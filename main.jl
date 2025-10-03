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

        # Lê pré-requisitos
        for i in 1:P
            precedence = split(strip(readline(io)))
            a = parse(Int, precedence[1])
            b = parse(Int, precedence[2])
            push!(get!(after, a, Int[]), b)
            push!(get!(before, b, Int[]), a)
        end

        return T, coords, before, after, P
    end
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
# FUNÇÃO: path_cost
# ------------------------------------------------------------------------------
# Calcula o custo total de uma ordenação linear de templos
#
# Parâmetros:
#   order - vetor com ordenação dos templos
#   coords - lista de tuplas de coordenadas
#
# Retorno:
#   cost - custo total do caminho (soma das distâncias entre templos consecutivos)
# ==============================================================================
function path_cost(order::Vector{Int}, coords)
    cost = 0
    for i in 1:(length(order)-1)
        t1 = order[i]
        t2 = order[i + 1]
        cost += dist_floor(coords[t1][1], coords[t1][2], coords[t2][1], coords[t2][2])
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
        if !is_valid(order, positions, left, right, before, after)
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
function differential_avaliation(current::Vector{Int}, candidate::Vector{Int}, left::Int, right::Int, current_cost::Int, coords)
    
    coords_T1 = coords[current[left]] # coords_T1 == coords[candidate[right]]
    coords_T2 = coords[current[right]] # coords_T2 == coords[candidate[left]]

    if right - left != 1
        coords_after_left = coords[current[left+1]] # coords_after_left == coords[candidate[left+1]]
        coords_before_right = coords[current[right-1]] # coords_before_right == coords[candidate[right-1]]
    
        old_distances = dist_floor(coords_T1[1], coords_T1[2], coords_after_left[1], coords_after_left[2]) + dist_floor(coords_T2[1], coords_T2[2], coords_before_right[1], coords_before_right[2])
        new_distances = dist_floor(coords_T2[1], coords_T2[2], coords_after_left[1], coords_after_left[2]) + dist_floor(coords_T1[1], coords_T1[2], coords_before_right[1], coords_before_right[2])
    else
        old_distances = 0
        new_distances = 0
    end

    # Se o índice mais à esquerda não for o primeiro, então considerar a distância entre left e left - 1
    if left > 1
        coords_before_left = coords[current[left-1]] # coords_before_left == coords[candidate[left-1]]

        old_distances += dist_floor(coords_T1[1], coords_T1[2], coords_before_left[1], coords_before_left[2])
        new_distances += dist_floor(coords_T2[1], coords_T2[2], coords_before_left[1], coords_before_left[2])
    end
    # Se o índice mais à direita não for o último, então considerar a distância entre right e right + 1
    if right < length(current)
        coords_after_right = coords[current[right+1]] # coords_after_right == coords[candidate[right+1]]

        old_distances += dist_floor(coords_T2[1], coords_T2[2], coords_after_right[1], coords_after_right[2])
        new_distances += dist_floor(coords_T1[1], coords_T1[2], coords_after_right[1], coords_after_right[2])
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
function lahc(coords, before, after, L::Int=50, max_iter::Int=10_000)
    T = length(coords)

    t_start = time()
    # Gera solução inicial gulosa
    #current = greedy_topological_order(T, after, D)
    current = simple_topological_order(T, after)
    current_cost = path_cost(current, coords)

    initial = copy(current)
    initial_cost = current_cost

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
            neighbor_cost = differential_avaliation(current, neighbor, left, right, current_cost, coords)

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

                 # Se current_cost < best_cost, atualiza o best
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
                current_cost = path_cost(current, coords)
                copyto!(neighbor, current) 
                copyto!(positions_neighbor, positions_current)

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

    t_end = time()
    duration = round(t_end - t_start, digits = 2)

    # Após as iterações acabarem, retornara melhor solução e seu custo
    return initial, initial_cost, best, best_cost, duration

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

    # Executa LAHC
    initial_order, initial_cost, best_order, best_cost, duration = lahc(coords, before, after, L, max_iter)

    println("="^50)
    println("PROBLEMA DE PEREGRINAÇÃO")
    println("="^50)
    println("Templos: $T")
    println("Pré-requisitos: $(precedences)")
    println("Semente: $seed")
    println("L: $L")
    println("Max Iter: $max_iter")
    println("-"^50)
    println("Custo inicial: $initial_cost")
    println("Custo final (LAHC): $best_cost")
    println("Melhoria: $(initial_cost - best_cost) ($(round((initial_cost - best_cost)/initial_cost*100, digits=2))%)")
    println("Tempo de execução: $duration segundos")
    println("="^50)
end

# Executa função principal
main()
