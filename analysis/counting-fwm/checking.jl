#!/usr/bin/env julia

"""
Frequency matching for FWM on a uniform channel grid.

Grid:
    f_k = f_ref + k * Δf

FWM convention:
    f_gen = f_a - f_b + f_c

Frequency matched to base channel i iff:
    a - b + c == i

The absolute reference frequency f_ref cancels on a uniform grid.
"""

struct MatchResult
    matched::Bool
    generated_index::Int
    target_index::Int
    mismatch_indices::Int
    mismatch_hz::Float64
end

function freq_matched(
    triplet::NTuple{3,Int},
    base::Int,
    spacing::Real;
    convention::Symbol = :abcstar,
)
    a, b, c = triplet

    generated_index =
        if convention == :abcstar
            # f_gen = f_a - f_b + f_c
            a - b + c
        elseif convention == :a_b_cstar
            # f_gen = f_a + f_b - f_c
            a + b - c
        else
            error("Unknown convention: $convention")
        end

    mismatch_indices = generated_index - base
    mismatch_hz = mismatch_indices * Float64(spacing)

    return MatchResult(
        mismatch_indices == 0,
        generated_index,
        base,
        mismatch_indices,
        mismatch_hz,
    )
end

function parse_triplet(s::String)
    parts = split(s, ",")
    length(parts) == 3 || error("Triplet must have form a,b,c")
    return Tuple(parse.(Int, parts))::NTuple{3,Int}
end

function main(args)
    if length(args) < 3
        println("""
        Usage:
            julia checking.jl a,b,c base spacing_Hz [convention]

        Examples:
            julia checking.jl 10,12,15 13 50e9
            julia checking.jl 10,12,15 13 50e9 abcstar
            julia checking.jl 10,12,15 13 50e9 a_b_cstar

        Conventions:
            abcstar    : f_gen = f_a - f_b + f_c
            a_b_cstar  : f_gen = f_a + f_b - f_c
        """)
        return
    end

    triplet = parse_triplet(args[1])
    base = parse(Int, args[2])
    spacing = parse(Float64, args[3])

    convention = length(args) ≥ 4 ? Symbol(args[4]) : :abcstar

    result = freq_matched(triplet, base, spacing; convention=convention)

    println("Triplet:              ", triplet)
    println("Base channel:         ", result.target_index)
    println("Generated channel:    ", result.generated_index)
    println("Mismatch in indices:  ", result.mismatch_indices)
    println("Mismatch in Hz:       ", result.mismatch_hz)
    println("Frequency matched:    ", result.matched)
end

main(ARGS)
