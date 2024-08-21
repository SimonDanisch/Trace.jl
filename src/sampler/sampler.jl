abstract type AbstractSampler end

mutable struct Sampler <: AbstractSampler
    samples_per_pixel::Int64
    current_pixel::Point2f
    current_pixel_sample_id::Int64

    samples_1d_array_sizes::Vector{Int32}
    samples_2d_array_sizes::Vector{Int32}

    sample_array_1d::Vector{Vector{Float32}}
    sample_array_2d::Vector{Vector{Point2f}}

    array_1d_offset::UInt64
    array_2d_offset::UInt64
end

function Sampler(samples_per_pixel::Integer)
    Sampler(
        samples_per_pixel, Point2f(-1), 1,
        Int32[], Int32[],
        Vector{Vector{Float32}}(undef, 0),
        Vector{Vector{Point2f}}(undef, 0),
        1, 1,
    )
end

using Random
using RandomNumbers.Xorshifts


const TRNG = Xoroshiro128Plus[]
# Reset the per-thread random seeds to make results reproducible
reseed!() = foreach(i-> Random.seed!(TRNG[i], i), 1:Threads.maxthreadid())

function __init__()
    # Instantiate 1 RNG (Random Number Generator) per thread, for performance.
    # This can't be done during precompilation since the number of threads isn't known then.
    resize!(TRNG, Threads.maxthreadid())
    for i in 1:Threads.nthreads()
        TRNG[i] = Xoroshiro128Plus(i)
    end
    nothing
end

"Per-thread rand()"
@inline function trand()
    @inbounds rng = TRNG[Threads.threadid()]
    rand(rng)
end

@inline function trand(::Type{T}) where {T}
    @inbounds rng = TRNG[Threads.threadid()]
    rand(rng, T)
end

function get_camera_sample(sampler::AbstractSampler, p_raster::Point2f)
    p_film = p_raster .+ get_2d(sampler)
    time = get_1d(sampler)
    p_lens = get_2d(sampler)
    CameraSample(p_film, p_lens, time)
end

@inline round_count(sampler::AbstractSampler, n::Integer) = n

"""
Other samplers are required to explicitly call this,
in their respective implementations.
"""
function start_pixel(sampler::Sampler, p::Point2f)
    sampler.array_1d_offset = sampler.array_2d_offset = 1
    sampler.current_pixel_sample_id = 1
    sampler.current_pixel = p
end

function start_next_sample(sampler::Sampler)
    sampler.array_1d_offset = sampler.array_2d_offset = 1
    sampler.current_pixel_sample_id += 1
    sampler.current_pixel_sample_id < sampler.samples_per_pixel
end

function set_sample_number(sampler::Sampler, sample_num::Integer)
    sampler.array_1d_offset = sampler.array_2d_offset = 1
    sampler.current_pixel_sample_id = sample_num
    sampler.current_pixel_sample_id < sampler.samples_per_pixel
end

function request_1d_array(sampler::Sampler, n::Integer)
    push!(sampler.samples_1d_array_sizes, n)
    push!(sampler.sample_array_1d, Vector{Float32}(undef, n * sampler.samples_per_pixel))
end

function request_2d_array(sampler::Sampler, n::Integer)
    push!(sampler.samples_2d_array_sizes, n)
    push!(sampler.sample_array_2d, Vector{Point2f}(undef, n * sampler.samples_per_pixel))
end

function get_1d_array(sampler::Sampler, n::Integer)
    sampler.array_1d_offset == length(sampler.sample_array_1d) + 1 && return nothing
    arr = @view sampler.sample_array_1d[sampler.array_1d_offset][sampler.current_pixel_sample_id*n:end]
    sampler.array_1d_offset += 1
    arr
end

function get_2d_array(sampler::Sampler, n::Integer)
    sampler.array_2d_offset == length(sampler.sample_array_2d) + 1 && return nothing
    arr = @view sampler.sample_array_2d[sampler.array_2d_offset][sampler.current_pixel_sample_id*n:end]
    sampler.array_2d_offset += 1
    arr
end

mutable struct PixelSampler <: AbstractSampler
    sampler::Sampler
    samples_1d::Vector{Vector{Float32}}
    samples_2d::Vector{Vector{Point2f}}
    current_1d_dimension::Int64
    current_2d_dimension::Int64
end

function PixelSampler(samples_per_pixel::Integer, n_sampled_dimensions::Integer)
    samples_1d = Vector{Vector{Float32}}(undef, n_sampled_dimensions)
    samples_2d = Vector{Vector{Point2f}}(undef, n_sampled_dimensions)
    for i in 1:n_sampled_dimensions
        samples_1d[i] = Vector{Float32}(undef, samples_per_pixel)
        samples_2d[i] = Vector{Point2f}(undef, samples_per_pixel)
    end
    PixelSampler(Sampler(samples_per_pixel), samples_1d, samples_2d, 1, 1)
end

start_pixel(p::PixelSampler, point::Point2f) = start_pixel(p.sampler, point)

function start_next_sample(ps::PixelSampler)
    ps.current_1d_dimension = ps.current_2d_dimension = 1
    start_next_sample(ps.sampler)
end

function set_sample_number(ps::PixelSampler, sample_num::Integer)::Bool
    ps.current_1d_dimension = ps.current_2d_dimension = 1
    set_sample_number(ps.sampler, sample_num)
end

function get_1d(ps::PixelSampler)
    ps.current_1d_dimension > length(ps.samples_1d) && return rand()
    v = ps.samples_1d[ps.current_1d_dimension][ps.sampler.current_pixel_sample_id]
    ps.current_1d_dimension += 1
    v
end

function get_2d(ps::PixelSampler)::Point2f
    ps.current_2d_dimension > length(ps.samples_2d) && return rand(Point2f)
    v = ps.samples_2d[ps.current_2d_dimension][ps.sampler.current_pixel_sample_id]
    ps.current_2d_dimension += 1
    v
end


struct UniformSampler <: AbstractSampler
    samples_per_pixel::Int64
end


@inline get_1d(::UniformSampler)::Float32 = rand(Float32)
@inline get_2d(::UniformSampler)::Point2f = rand(Point2f)

# include("stratified.jl")
