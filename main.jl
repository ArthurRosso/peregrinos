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
#   precedences - vetor de tuplas com pré-requisitos (a,b)
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

        # Lê pré-requisitos
        precedences = Vector{Tuple{Int,Int}}(undef, P)
        for i in 1:P
            parts = split(strip(readline(io)))
            a = parse(Int, parts[1])
            b = parse(Int, parts[2])
            precedences[i] = (a, b)
        end

        return T, coords, precedences
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
#   precedences - vetor de pré-requisitos
#
# Retorno:
#   order - vetor com ordenação topológica dos templos
# ==============================================================================
function simple_topological_order(T, precedences)
    # Construir lista de adjacência e graus de entrada
    adj = [Int[] for _ in 1:T]
    indeg = zeros(Int, T)
    for (a,b) in precedences
        push!(adj[a], b)
        indeg[b] += 1
    end

    # Fila com nós sem predecessores
    queue = [i for i in 1:T if indeg[i] == 0]
    order = Int[]

    while !isempty(queue)
        # Seleciona sempre o menor índice (determinístico)
        v = popfirst!(queue)
        push!(order, v)

        # Atualiza graus dos vizinhos
        for w in adj[v]
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
#   precedences - vetor de pré-requisitos
#   D - matriz de distâncias
#
# Retorno:
#   order - ordenação topológica gulosa dos templos
# ==============================================================================
function greedy_topological_order(T::Int, precedences::Vector{Tuple{Int,Int}}, D::Array{Int,2})
    # Construir grafo de precedências
    adj = [Int[] for _ in 1:T]
    indeg = zeros(Int, T)
    for (a,b) in precedences
        push!(adj[a], b)
        indeg[b] += 1
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
    for w in adj[first]
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
        for w in adj[best]
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
# FUNÇÃO: differential_avaliation
# ------------------------------------------------------------------------------
# Realiza a avaliação diferencial para verificar o custo de uma solução vizinha
#
# Parâmetros:
#   current - vetor atual com ordenação dos templos
#   candidate - vetor candidato
#   i - um dos índices escolhidos para swap na função random_neighbor
#   j - o outro índice escolhido para swap
#   current_cost - custo da solução atual
#   D - matriz de distâncias
#
# Retorno:
#   o custo do candidato
# ==============================================================================
function differential_avaliation(current::Vector{Int}, candidate::Vector{Int}, i::Int, j::Int, current_cost::Int, D::Matrix{Int})
    
    # Verifica qual é o índice mais à esquerda e qual é o índice mais à direita
    if i < j
        left = i
        right = j
    else 
        left = j
        right = i 
    end 

    old_distances = D[current[left], current[left + 1]] + D[current[right], current[right -1]] 
    new_distances = D[candidate[left], candidate[left + 1]] + D[candidate[right], candidate[right -1]]

    # Se o índice mais à esquerda não for o primeiro, então considerar a distância entre left e left - 1
    if left > 1
        old_distances += D[current[left], current[left - 1]]
        new_distances += D[candidate[left], candidate[left - 1]]
    end
    # Se o índice mais à direita não for o último, então considerar a distância entre right e right + 1
    if right < length(current)
        old_distances += D[current[right], current[right + 1]]
        new_distances += D[candidate[right], candidate[right + 1]]
    end

    # A avaliação diferencial consiste em subtrair as distâncias antigas e somar as novas
    return current_cost - old_distances + new_distances


# ==============================================================================
# FUNÇÃO: random_neighbor
# ------------------------------------------------------------------------------
# Gera um vizinho aleatório trocando dois templos na ordenação
# A troca só é mantida se não violar pré-requisitos
#
# Parâmetros:
#   order - ordenação atual
#   precedences - vetor de pré-requisitos
#
# Retorno:
#   new_order - nova ordenação (pode ser igual à original se troca for inválida)
# ==============================================================================
function random_neighbor(order::Vector{Int}, precedences)
    T = length(order)
    new_order = copy(order)

    # Seleciona dois índices aleatórios distintos
    i, j = rand(1:T), rand(1:T)
    while i == j
        j = rand(1:T)
    end

    # Troca os templos nas posições i e j
    new_order[i], new_order[j] = new_order[j], new_order[i]

    # Verifica se a nova ordenação respeita todos os pré-requisitos
    pos = Dict(v => idx for (idx,v) in enumerate(new_order))
    for (a,b) in precedences
        if pos[a] >= pos[b]
            return order, i, j  # Devolve solução original se violar pré-requisitos
        end
    end

    return new_order, i, j
end

# ==============================================================================
# FUNÇÃO: lahc
# ------------------------------------------------------------------------------
# Implementação do algoritmo Late Acceptance Hill Climbing (LAHC)
#
# Parâmetros:
#   D - matriz de distâncias T×T
#   precedences - vetor de pares (a,b) representando pré-requisitos
#   L - tamanho da lista de aceitação (default: 50)
#   max_iter - número máximo de iterações (default: 10_000)
#
# Retorno:
#   best - melhor ordenação encontrada
#   best_cost - custo da melhor ordenação
# ==============================================================================
function lahc(D::Matrix{Int}, precedences, L::Int=50, max_iter::Int=10_000)
    T = size(D,1)

    # Gera solução inicial gulosa
    current = greedy_topological_order(T, precedences, D)
    current_cost = path_cost(current, D)

    best = copy(current)
    best_cost = current_cost

    # Inicializa lista de aceitação com custo atual
    costs = fill(current_cost, L)

    # Loop principal do LAHC
    for it in 1:max_iter
        # Gera vizinho aleatório
        candidate, i, j = random_neighbor(current, precedences)
        candidate_cost = differential_avaliation(current, candidate, i, j, current_cost, D)

        # Índice circular na lista de aceitação
        idx = (it % L) + 1

        # Regra de aceitação do LAHC
        if candidate_cost <= current_cost || candidate_cost <= costs[idx]
            current = candidate
            current_cost = candidate_cost
        end

        # Atualiza lista de aceitação
        costs[idx] = current_cost

        # Atualiza melhor solução global
        if current_cost < best_cost
            best = copy(current)
            best_cost = current_cost
        end
    end

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
    T, coords, precedences = read_instance(path)
    D = build_dist_matrix(T, coords)

    # Gera e avalia solução inicial
    order = greedy_topological_order(T, precedences, D)
    cost = path_cost(order, D)

    println("="^50)
    println("PROBLEMA DE PEREGRINAÇÃO")
    println("="^50)
    println("Templos: $T")
    println("Pré-requisitos: $(length(precedences))")
    println("Semente: $seed")
    println("L: $L")
    println("Max Iter: $max_iter")
    println("-"^50)
    println("Custo inicial: $cost")

    # Executa LAHC
    best_order, best_cost = lahc(D, precedences, L, max_iter)

    println("Custo final (LAHC): $best_cost")
    println("Melhoria: $(cost - best_cost) ($(round((cost - best_cost)/cost*100, digits=2))%)")
    println("="^50)
end

# Executa função principal
main()
