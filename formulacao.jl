using JuMP
using HiGHS

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

        return T, P, coords, precedences
    end
end


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


function formulation(T::Int, P::Int, dist::Matrix{Int}, precedences::Vector{Tuple{Int, Int}}, time_limit::Float64, seed::Int)
    
    model = Model(HiGHS.Optimizer)

    # Configurações do otimizador:
    set_attribute(model, "time_limit", time_limit)
    set_attribute(model, "random_seed", seed)

    # Variáveis que indicam se o arco (u, v) foi usado
    @variable(model, x[u=1:T, v=1:T], Bin)

    # Variáveis que indicam a ordem de uso de cada arco (u,v).
    # Se (u, v) foi usado, então y(u, v) > 0, senão y(u, v) == 0
    @variable(model, y[u=1:T, v=1:T] >= 0)

    # Variáveis que indicam se o templo u é o primeiro
    @variable(model, starting_point[u=1:T], Bin)

    # Variáveis que indicam se o templo u é o último
    @variable(model, endpoint[u=1:T], Bin)

    @objective(model, Min, sum(dist[u, v] * x[u, v] for u in 1:T for v in 1:T))

    # RESTRIÇÃO 1:
    # Apenas um templo pode ser o inicial, então a soma das variáveis
    # que indicam o templo de início deve ser igual a 1
    @constraint(model, [t in T], sum(starting_point[t]) == 1)

    # RESTRIÇÃO 2:
    # Apenas um templo pode ser o final, então a soma das variáveis
    # que indicam o templo final deve ser igual a 1
    @constraint(model, [t in T], sum(endpoint[t]) == 1)

    # RESTRIÇÃO 3:
    # Não é possível que um templo seja inicial e final ao mesmo tempo
    @constraint(model, [t in T], starting_point[t] + endpoint[t] <= 1)

    # RESTRIÇÃO 4:
    # Todo templo, exceto o inicial, deve ter um arco que incide nele
    @constraint(model, [v in T], starting_point[v] + sum(x[u, v] for u in T) == 1)

    # RESTRIÇÃO 5:
    # Todo templo, exceto o final, deve ter um arco que parte dele
    @constraint(model, [v in T], endpoint[v] + sum(x[v, u] for u in T) == 1)

    # RESTRIÇÃO 6:
    # Se o arco (u, v) é usado, então a ordem dele é um valor maior
    # que zero. Caso contrário, o valor pode ser 0
    @constraint(model, [u in T, v in T], y[u, v] >= x[u, v])

    # RESTRIÇÃO 7:
    # Se o arco (u, v) é usado, então a ordem dele é um valor no 
    # máximo igual a T - 1. Caso contrário, é no máximo 0
    @constraint(model, [u in T, v in T], y[u, v] <= (T -1) * x[u, v])

    #OBS: juntas, as restrições 6 e 7 garantem que, para cada templo
    # v, há no máximo 1 valor y[u, v] maior que 0 e no máximo 1 valor
    # y[v, w] maior que 0. Isso é importante para a restrição 8

    # RESTRIÇÃO 8:
    # Para  cada templo v, a ordem da aresta que inside nele escolhida
    # para compor o caminho deve ser 1 unidade menor que a ordem da 
    # aresta que parte dele escolhida para compor o caminho.
    @constraint(model, [v in T], sum(y[u,v] for u in T) + 1 == sum(y[v, w] for w in T) + (T -1) * endpoint[v])

    # RESTRIÇÃO 9:
    # Garante que a ordem dos pré-requisitos é respeitada
    for (u, v) in precedences
        @constraint(model, sum(y[u, s] for s in T) <= sum(y[t, v] for t in T))
    end

    optimize!(model)

    status = termination_status(model)
    achived_time_limit = (status == MOI.TIME_LIMIT)
    found_optimum = (status == MOI.TIME_LIMIT)

    best_solution = Inf
    if has_values(model)
        best_solution = objective_value(model)
    end

    time = solve_time(model)

    return best_solution, found_optimum, achived_time_limit, time

end

function main()

    # Verifica argumentos
    if length(ARGS) < 1
        println("Uso: julia formulacao.jl caminho/para/instancia.txt [tempo_limite] [seed]")
        return
    end

    path = ARGS[1]

    # Configura parâmetros opcionais
    time_limit = length(ARGS) > 1 ? parse(Float64, ARGS[2]) : 5
    seed = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 1
    
    # Lê a instância
    T, P, coords, precedences = read_instance(path)

    # Constrói a matriz de distâncias
    dist = build_dist_matrix(T, coords)

    # Executa a formulação
    best_solution, found_optimum, achived_time_limit, time = formulation(T, P, dist, precedences, time_limit, seed)

    # Impressão dos resultados:
    println("="^50)
    println("PROBLEMA DE PEREGRINAÇÃO")
    println("="^50)
    println("Templos: $T")
    println("Pré-requisitos: $P")
    println("Tempo limite: $time_limit")
    println("Semente: $seed")
    println("-"^50)
    println("Melhor solução: $best_solution")
    println("Solução ótima? $found_optimum")
    println("Tempo de execução: $time")
    println("Tempo limite atingido? $achived_time_limit")
    println("="^50)
end

main()



