#!/usr/bin/env julia
# ==============================================================================
# Metaheurística: Late Acceptance Hill Climbing (LAHC) para Problema de Peregrinação
#
# Uso:
#   julia initial_solution.jl caminho/para/instancia.txt [L] [max_iter] [seed]
#
# Parâmetros:
#   ARGS[1] - caminho para o arquivo da instância
#   ARGS[2] - max_iter (opcional): número máximo de iterações (default: 1000000)
#   ARGS[3] - seed (opcional): semente para aleatoriedade (default: 42)
#   ARGS[4] - L (opcional): tamanho da lista de aceitação (default: 1000)
#
# Formato da instância:
#   T (número de templos)
#   T linhas com coordenadas (x y)
#   P (número de pré-requisitos)
#   P linhas com pré-requisitos (a b) onde a deve vir antes de b
# ==============================================================================

using Random

# ==============================================================================
# FUNÇÃO: dist_floor
# ------------------------------------------------------------------------------
# Calcula a distância euclidiana entre dois pontos e arredonda para baixo
#
# Parâmetros:
#   x1, y1 - coordenadas do primeiro ponto
#   x2, y2 - coordenadas do segundo ponto
#
# Retorno:
#   Distância euclidiana arredondada para baixo (multiplicada por 100 para precisão)
# ==============================================================================
function dist_floor(x1::Int, y1::Int, x2::Int, y2::Int)::Int
    dx = x1 - x2
    dy = y1 - y2
    return floor(Int, sqrt(dx^2 + dy^2)*100)
end

# ==============================================================================
# FUNÇÃO: read_instance
# ------------------------------------------------------------------------------
# Lê uma instância do problema de peregrinação a partir de um arquivo
#
# Parâmetros:
#   path - caminho para o arquivo da instância
#
# Retorno:
#   T - número de templos
#   coords - vetor de tuplas com coordenadas (x,y) dos templos
#   before - dicionário que associa cada templo i com um vetor de templos que devem ser visitados antes de i
#   after - dicionário que associa cada templo i com um vetor de templos que devem ser visitados depois de i
#   precedences - o número de pré-requisitos
# ==============================================================================
function read_instance(path::String)
    open(path, "r") do io
        # Lê número de templos
        line = readline(io)
        T = parse(Int, strip(line))

        # Lê coordenadas dos templos
        coords = Vector{Tuple{Int,Int}}(undef, T)
        for i in 1:T
            parts = split(strip(readline(io)))
            coords[i] = (parse(Int, parts[1]), parse(Int, parts[2]))
        end

        # Lê número de pré-requisitos
        P = parse(Int, strip(readline(io)))

        # Dicionário em que, apara cada templo i, retorna um vetor contendo os templos que devem 
        # ser visitados depois de i
        after = Dict{Int, Vector{Int}}()

        # Dicionário em que, apara cada templo i, retorna um vetor contendo os templos que devem 
        # ser visitados antes de i
        before = Dict{Int, Vector{Int}}()

        precedences = 0
        # Lê pré-requisitos
        for i in 1:P
            precedence = split(strip(readline(io)))
            a = parse(Int, precedence[1])
            b = parse(Int, precedence[2])
            push!(get!(after, a, Int[]), b)
            push!(get!(before, b, Int[]), a)
            precedences += 1
        end

        return T, coords, before, after, precedences
    end
end

# ==============================================================================
# FUNÇÃO: build_dist_matrix
# ------------------------------------------------------------------------------
# Constrói a matriz de distâncias entre todos os pares de templos
#
# Parâmetros:
#   T - número de templos
#   coords - vetor de coordenadas dos templos
#
# Retorno:
#   D - matriz T×T com distâncias entre templos
# ==============================================================================
function build_dist_matrix(T::Int, coords::Vector{Tuple{Int,Int}})
    D = Array{Int}(undef, T, T)
    for i in 1:T
        xi, yi = coords[i]
        for j in 1:T
            xj, yj = coords[j]
            D[i,j] = dist_floor(xi, yi, xj, yj)
        end
    end
    return D
end

# ==============================================================================
# FUNÇÃO: simple_topological_order
# ------------------------------------------------------------------------------
# Gera uma ordenação topológica simples e determinística dos templos
# usando o algoritmo de Kahn com seleção pelo menor índice
#
# Parâmetros:
#   T - número de templos
#   after - dicionário que associa cada templo i com um vetor de templos que devem ser visitados depois de i
#
# Retorno:
#   order - vetor com ordenação topológica dos templos
# ==============================================================================
function simple_topological_order(T, after)
    # Construir lista com graus de adjacência (o número de templos que devem ser visitados antes
    # do templo i)
    indeg = zeros(Int, T)
    for vector in values(after)
        for temple in vector
            indeg[temple] += 1
        end
    end

    # Fila com templos sem predecessores
    queue = [i for i in 1:T if indeg[i] == 0]
    order = Int[]

    while !isempty(queue)
        # Seleciona sempre o menor índice (determinístico)
        v = popfirst!(queue)
        push!(order, v)

        # Atualiza graus dos vizinhos
        for w in get(after, v, Int[])
            indeg[w] -= 1
            if indeg[w] == 0
                push!(queue, w)
            end
        end
        sort!(queue) # Mantém ordem determinística
    end

    # Verifica se a ordenação é completa
    if length(order) != T
        error("Instância tem ciclo de precedência — sem ordem topológica")
    end

    return order
end

# ==============================================================================
# FUNÇÃO: choose_start
# ------------------------------------------------------------------------------
# Escolhe o nó inicial entre os disponíveis baseado na soma das distâncias
#
# Parâmetros:
#   available - vetor de nós disponíveis
#   D - matriz de distâncias
#
# Retorno:
#   best - nó com menor soma de distâncias para todos os outros
# ==============================================================================
function choose_start(available::Vector{Int}, D::Array{Int,2})
    best = available[1]
    bestscore = sum(D[best, :])
    for v in available
        s = sum(D[v, :])
        if s < bestscore || (s == bestscore && v < best)
            best = v
            bestscore = s
        end
    end
    return best
end

# ==============================================================================
# FUNÇÃO: greedy_topological_order
# ------------------------------------------------------------------------------
# Gera uma ordenação topológica gulosa considerando distâncias entre templos
# Usa algoritmo de Kahn com seleção gulosa baseada na distância ao último templo
#
# Parâmetros:
#   T - número de templos
#   after - dicionário que associa cada templo i com um vetor de templos que devem ser visitados depois de i
#   D - matriz de distâncias
#
# Retorno:
#   order - ordenação topológica gulosa dos templos
# ==============================================================================
function greedy_topological_order(T::Int, after, D::Array{Int,2})
    # Construir grafo de precedências
    indeg = zeros(Int, T)
    for vector in values(after)
        for temple in vector
            indeg[temple] += 1
        end
    end

    # Conjunto de nós disponíveis (grau de entrada zero)
    available = Int[]
    for v in 1:T
        if indeg[v] == 0
            push!(available, v)
        end
    end
    sort!(available)  # Ordem determinística inicial

    order = Int[]

    # Escolhe primeiro nó baseado na soma de distâncias
    if isempty(available)
        error("Instância inválida: ciclo de precedências detectado")
    end
    first = choose_start(available, D)
    deleteat!(available, findfirst(==(first), available))
    push!(order, first)
    last = first

    # Atualiza graus dos vizinhos do primeiro nó
    for w in get(after, first, Int[])
        indeg[w] -= 1
        if indeg[w] == 0
            push!(available, w)
        end
    end
    sort!(available)

    # Loop principal: seleção gulosa baseada na distância ao último templo
    while length(order) < T
        if isempty(available)
            error("Ciclo detectado: impossível completar ordenação topológica")
        end

        # Escolhe o nó disponível mais próximo do último templo
        best = available[1]
        bestdist = D[last, best]
        for v in available
            d = D[last, v]
            if d < bestdist || (d == bestdist && v < best)
                best = v
                bestdist = d
            end
        end

        # Adiciona nó selecionado à ordenação
        deleteat!(available, findfirst(==(best), available))
        push!(order, best)
        last = best

        # Atualiza graus dos vizinhos
        for w in get(after, best, Int[])
            indeg[w] -= 1
            if indeg[w] == 0
                push!(available, w)
            end
        end
        sort!(available)
    end

    return order
end

# ==============================================================================
# FUNÇÃO: path_cost
# ------------------------------------------------------------------------------
# Calcula o custo total de uma ordenação linear de templos
#
# Parâmetros:
#   order - vetor com ordenação dos templos
#   D - matriz de distâncias
#
# Retorno:
#   cost - custo total do caminho (soma das distâncias entre templos consecutivos)
# ==============================================================================
function path_cost(order::Vector{Int}, D::Matrix{Int})
    cost = 0
    for i in 1:(length(order)-1)
        cost += D[order[i], order[i+1]]
    end
    return cost
end

# ==============================================================================
# FUNÇÃO: is_valid
# ------------------------------------------------------------------------------
# Verifica se uma ordenação é válida. Para isso, analisa se todos os templos que devem ser
# visitados antes de left estão à esquerda de left, e se todos os templos que devem ser visitados
# depois de right à direita de right. Presume-se que que todos os templos que devem ser visitados 
# depois de left estão à direita de left, e que todos os templos que devem ser visitados antes de
# right estão à esquerda de right
#
#
# Parâmetros:
#   order - vetor com ordenação dos templos
#   positions - vetor que indica qual posição o elemento i ocupa no vetor order
#   left - índice da posição esquerda da troca
#   right - índice da posição direita da troca
#   before - dicionário que associa cada templo i com um vetor de templos que devem ser visitados antes de i
#   after - dicionário que associa cada templo i com um vetor de templos que devem ser depois antes de i
#
# Retorno:
#   true se a ordenação é válida, false caso contrário
# ==============================================================================
function is_valid(order, positions, left, right, before, after)

    temple_left = order[left]
    temple_right = order[right]

    # Templos que devem vir antes de temple_left
    for temple in get(before, temple_left, Int[])
        if positions[temple] > positions[temple_left]
            return false
        end
    end

    # Templos que devem vir depois de temple_right
    for temple in get(after, temple_right, Int[])
        if positions[temple] < positions[temple_right]
                return false
        end
        break
    end

    return true
end


# ==============================================================================
# FUNÇÃO: shake
# ------------------------------------------------------------------------------
# Produz uma solução tentando realizar 10000 swaps de duas posições aleatórias 
# de uma solução a fim de produzir uma nova solução válida
#
# Parâmetros:
#   order - ordenação atual
#   before - dicionário que associa cada templo i com um vetor de templos que devem ser visitados antes de i
#   after - dicionário que associa cada templo i com um vetor de templos que devem ser depois antes de i
#
# Retorno:
#   new_order - nova ordenação
# ==============================================================================
function shake(order, positions, before, after)
    
    T = length(order)

    for i in 1:10000
        j, k = rand(1:T), rand(1:T)
        while j == k
            k = rand(1:T)
        end

        if j < k
            left = j
            right = k
        else
            left = k
            right = j
        end

        # Troca os templos nas posições left e right
        order[right], order[left] = order[left], order[right]
        positions[order[left]], positions[order[right]] = left, right

        # Se a troca for inválida, então desfazer a troca
        if !is_valid(order, left, right, before, after)
            order[right], order[left] = order[left], order[right]
            positions[order[left]], positions[order[right]] = left, right
        end

    end

end

# ==============================================================================
# FUNÇÃO: print_status
# ------------------------------------------------------------------------------
# Imprime no terminal o tempo desde o início da execução do algoritmo, o valor da
# solução atual e uma representação da solução atual
#
# Parâmetros:
#   t_start - tempo de início
#   order - ordenação atual
#   cost - custo da ordenação
#
# ==============================================================================
function print_status(t_start, order, cost)
    t_now = time()
    t = round(t_now - t_start, digits = 2)
    println("-"^50)
    println("Tempo: $t segundos")
    println("Custo: $cost")
    T = length(order)
    for i in 1:(T-1)
        print("$(order[i]) -> ")
    end
    println(order[T])
end

# ==============================================================================
# FUNÇÃO: differential_avaliation
# ------------------------------------------------------------------------------
# Realiza a avaliação diferencial para verificar o custo de uma solução vizinha
#
# Parâmetros:
#   current - vetor atual com ordenação dos templos
#   candidate - vetor candidato
#   left - o índice modificado da esquerda
#   right - o índice modificado da direta
#   current_cost - custo da solução atual
#   D - matriz de distâncias
#
# Retorno:
#   o custo do candidato
# ==============================================================================
function differential_avaliation(current::Vector{Int}, candidate::Vector{Int}, left::Int, right::Int, current_cost::Int, D::Matrix{Int})
    
    old_distances = D[current[left], current[left+1]] + D[current[right], current[right-1]] 
    new_distances = D[candidate[left], candidate[left+1]] + D[candidate[right], candidate[right-1]]

    # Se o índice mais à esquerda não for o primeiro, então considerar a distância entre left e left - 1
    if left > 1
        old_distances += D[current[left], current[left-1]]
        new_distances += D[candidate[left], candidate[left-1]]
    end
    # Se o índice mais à direita não for o último, então considerar a distância entre right e right + 1
    if right < length(current)
        old_distances += D[current[right], current[right+1]]
        new_distances += D[candidate[right], candidate[right+1]]
    end

    # A avaliação diferencial consiste em subtrair as distâncias antigas e somar as novas
    return current_cost - old_distances + new_distances
end

# ==============================================================================
# FUNÇÃO: lahc
# ------------------------------------------------------------------------------
# Implementação do algoritmo Late Acceptance Hill Climbing (LAHC)
#
# Parâmetros:
#   D - matriz de distâncias T×T
#   before - dicionário que associa cada templo i com um vetor de templos que devem ser visitados antes de i
#   after - dicionário que associa cada templo i com um vetor de templos que devem ser visitados depois de i
#   L - tamanho da lista de aceitação (default: 50)
#   max_iter - número máximo de iterações (default: 10_000)
#
# Retorno:
#   best - melhor ordenação encontrada
#   best_cost - custo da melhor ordenação
# ==============================================================================
function lahc(D::Matrix{Int}, before, after, L::Int=50, max_iter::Int=10_000)
    T = size(D,1)

    t_start = time()
    # Gera solução inicial gulosa
    #current = greedy_topological_order(T, after, D)
    current = simple_topological_order(T, after)
    current_cost = path_cost(current, D)

    positions_current = Vector{Int}(undef, T) # vetor auxiliar que serve para indicar a posição de cada templo em uma ordem
    for i in 1:T
        positions_current[current[i]] = i
    end

    best = copy(current)
    best_cost = current_cost

    print_status(t_start, current, current_cost)

    # Inicializa lista de aceitação com custo atual
    costs = fill(current_cost, L)

    it = max_iter # Controla a quantidade de iterações
    neighbor = copy(current) 
    positions_neighbor = copy(positions_current)
    i_costs = 1 # variável que itera pelo histórico de custos
    neighborhood = collect(1:T) # vetor auxiliar para percorrer a vizinhança
    shuffle!(neighborhood)
    first_position = 1
    second_position = 2
    while it != 0

        # Geração de um vizinho por swap entre dois templos nas pos
        left = neighborhood[first_position]
        right = neighborhood[second_position]

        if left > right
            right, left = left, right
        end 
        neighbor[right], neighbor[left] = neighbor[left], neighbor[right]
        # Atualiza o vetor de posições do vizinho
        positions_neighbor[neighbor[left]], positions_neighbor[neighbor[right]] = left, right
        
        accepted = false

        # Decremento das iterações (as iterações são referentes à geração de vizinhos, 
        # tanto inválidos quanto válidos)
        it -= 1

        # Se o vizinho é válido, fazer os testes do LAHC
        if is_valid(neighbor, positions_neighbor, left, right, before, after)
            #Cálculo do custo do vizinho
            neighbor_cost = differential_avaliation(current, neighbor, left, right, current_cost, D)

            # Se for aceito na regra de aceitação do LAHC
            if neighbor_cost < current_cost || neighbor_cost < costs[i_costs]
                
                accepted = true
                
                # Atualiza o current
                current[right], current[left] = current[left], current[right]
                positions_current[current[left]], positions_current[current[right]] = left, right
                current_cost = neighbor_cost

                # Atualiza o histórico
                costs[i_costs] = current_cost
                i_costs = (i_costs % L) + 1

                # Reinicia a vizinhança
                shuffle!(neighborhood)
                first_position = 1
                second_position = 2

                #OBS: não é necessário fazer copyto!(neighbor, current), uma vez que neighbor == current

                 # Se currenct_cost < best_cost, atualiza o best
                if current_cost < best_cost
                    copyto!(best, current)
                    best_cost = current_cost
                    print_status(t_start, best, best_cost)
                end
            end
        end

        # Se o vizinho não foi aceito (por ser inválido ou por falhar na regra do LAHC),
        # preparamos para gerar um novo vizinho
        if !accepted
                
            # Se first_position < T - 1, então a vizinhança de current não acabou
            if first_position < T - 1
                # Desfaz o vizinho (na prática, neighbor volta a ser igual a current)
                neighbor[right], neighbor[left] = neighbor[left], neighbor[right]
                positions_neighbor[neighbor[left]], positions_neighbor[neighbor[right]] = left, right

                # Se second_position < T, então second_position pode ser incrementado 
                if second_position < T
                    second_position += 1
                # Se second position == T, então first_position deve ser incrementado e second_position = first_position + 1
                else
                    first_position += 1
                    second_position = first_position + 1
                end
            # Se first_position == T - 1, então a vizinhança de current acabou
            else
                # Como a vizinhança de current acabou, current é um mínimo local. 
                # Para escapar desse mínimo, vamos fazer um shake
                shake(current, positions_current, before, after)
                current_cost = path_cost(current, D)
                copyto!(neighbor, current) 
                copyto!(positions_neighbor, positions_current)
                println("shake")

                # Se current_cost < best_cost, atualiza o best
                if current_cost < best_cost
                    copyto!(best, current)
                    best_cost = current_cost
                    print_status(t_start, best, best_cost)
                end

                # Reinicia o histórico
                costs[i_costs] = current_cost
                i_costs = 1
                costs = fill(current_cost, L)

                # Reinicia a vizinhança
                shuffle!(neighborhood)
                first_position = 1
                second_position = 2
            end
        end
            
    end

    # Após as iterações acabarem, retornara melhor solução e seu custo
    return best, best_cost

end

# ==============================================================================
# FUNÇÃO: main
# ------------------------------------------------------------------------------
# Função principal do programa
# Lê instância, executa algoritmos e exibe resultados
# ==============================================================================
function main()
    # Verifica argumentos
    if length(ARGS) < 1
        println("Uso: julia initial_solution.jl caminho/para/instancia.txt [max_iter] [seed] [L]")
        return
    end

    path = ARGS[1]

    # Configura parâmetros opcionais
    max_iter = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1000000
    seed = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1
    L = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 1000

    # Configura semente aleatória
    Random.seed!(seed)

    # Lê e processa instância
    T, coords, before, after, precedences = read_instance(path)
    D = build_dist_matrix(T, coords)

    # Gera e avalia solução inicial
    #order = greedy_topological_order(T, after, D)
    order = simple_topological_order(T, after)
    cost = path_cost(order, D)

    # Executa LAHC
    best_order, best_cost = lahc(D, before, after, L, max_iter)

    println("="^50)
    println("PROBLEMA DE PEREGRINAÇÃO")
    println("="^50)
    println("Templos: $T")
    println("Pré-requisitos: $(precedences)")
    println("Semente: $seed")
    println("L: $L")
    println("Max Iter: $max_iter")
    println("-"^50)
    println("Custo inicial: $cost")
    println("Custo final (LAHC): $best_cost")
    println("Melhoria: $(cost - best_cost) ($(round((cost - best_cost)/cost*100, digits=2))%)")
    println("="^50)
end

# Executa função principal
main()
