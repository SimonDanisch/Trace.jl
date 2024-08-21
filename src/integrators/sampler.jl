abstract type SamplerIntegrator <: Integrator end

struct WhittedIntegrator{C<: Camera, S <: AbstractSampler} <: SamplerIntegrator
    camera::C
    sampler::S
    max_depth::Int64
end

function sample_pixel(i::A, scene::B, t_sampler::C, camera::F, pixel::G, spp_sqr::H) where {A, B, C, F, G, H}
    l = RGBSpectrum(0f0)
    for _ in 1:t_sampler.samples_per_pixel
        camera_sample = get_camera_sample(t_sampler, pixel)
        ray, ω = generate_ray_differential(camera, camera_sample)
        ray = scale_differentials(ray, spp_sqr)
        if ω > 0.0f0
            l = li(i, ray, scene, 1)
        end
    end
    return isnan(l) ? RGBSpectrum(0f0) : l
end

@noinline function sample_kernel(i::A, camera::B, scene::C, film::D, film_tile::E, tile_bounds::F)
    t_sampler = i.sampler
    spp_sqr = 1f0 / √Float32(t_sampler.samples_per_pixel)
    f_xyz = film.pixels.xyz
    for pixel in tile_bounds
        l = sample_pixel(i, scene, t_sampler, film, film_tile, camera, pixel, spp_sqr)
        idx = get_pixel_index(film, pixel)
        if checkbounds(Bool, f_xyz, idx)
            f_xyz[idx] += to_XYZ(l)
        end
    end
end


function sampling_kernel(i::A, scene::B, sampler::C, camera::D, f_xyz::E, spp_sqr::F) where {A, B, C, D, E, F}
    Threads.@threads for idx in CartesianIndices(size(f_xyz))
        pixel = Point2f(Tuple(idx))
        l = sample_pixel(i, scene, sampler, camera, pixel, spp_sqr)
        f_xyz[idx] += to_XYZ(l)
    end
end

"""
Render scene.
"""
function (i::SamplerIntegrator)(scene::Scene, film::Film)
    # TODO visualize tile bounds to see if they overlap
    film = get_film(i.camera)
    f_xyz = film.pixels.xyz
    bar = Progress(length(f_xyz), 1)
    @info "Utilizing $(Threads.nthreads()) threads"
    camera = i.camera
    sampler = i.sampler
    spp_sqr = 1.0f0 / √Float32(sampler.samples_per_pixel)
    sampling_kernel(i, scene, sampler, camera, f_xyz, spp_sqr)
    save(film)
end

function li(
        i::WhittedIntegrator, ray::RayDifferentials, scene::Scene, depth::Int64,
    )::RGBSpectrum

    l = RGBSpectrum(0f0)
    # Find closest ray intersection or return background radiance.
    hit, primitive, si = intersect!(scene, ray)
    if !hit
        for light in scene.lights
            l += le(light, ray)
        end
        return l
    end
    # Compute emmited & reflected light at ray intersection point.
    # Initialize common variables for Whitted integrator.
    core = si.core
    n = si.shading.n
    wo = core.wo
    # Compute scattering functions for surface interaction.
    si, bsdf = compute_scattering!(primitive, si, ray)
    if isnan(bsdf.η) # TODO, better way to return "nothing" BSDFs
        return li(
            i, RayDifferentials(spawn_ray(si, ray.d)),
            scene, depth,
        )
    end
    # Compute emitted light if ray hit an area light source.
    l += le(si, wo)
    # Add contribution of each light source.
    for light in scene.lights
        sampled_li, wi, pdf, visibility_tester = sample_li(
            light, core, get_2d(i.sampler),
        )
        (is_black(sampled_li) || pdf ≈ 0f0) && continue
        f = bsdf(wo, wi)
        if !is_black(f) && unoccluded(visibility_tester, scene)
            l += f * sampled_li * abs(wi ⋅ n) / pdf
        end
    end
    if depth + 1 ≤ i.max_depth
        # Trace rays for specular reflection & refraction.
        l += specular_reflect(bsdf, i, ray, si, scene, depth)
        l += specular_transmit(bsdf, i, ray, si, scene, depth)
    end
    l
end

function specular_reflect(
        bsdf, i::I, ray::RayDifferentials,
        surface_intersect::SurfaceInteraction, scene::Scene, depth::Int64,
    ) where I<:SamplerIntegrator

    # Compute specular reflection direction `wi` and BSDF value.

    wo = surface_intersect.core.wo
    type = BSDF_REFLECTION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        bsdf, wo, get_2d(i.sampler), type,
    )
    # Return contribution of specular reflection.
    ns = surface_intersect.shading.n
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        return RGBSpectrum(0f0)
    end
    # Compute ray differential for specular reflection.
    rd = RayDifferentials(spawn_ray(surface_intersect, wi))
    if ray.has_differentials
        rx_origin = surface_intersect.core.p + surface_intersect.∂p∂x
        ry_origin = surface_intersect.core.p + surface_intersect.∂p∂y
        # Compute differential reflected directions.
        ∂n∂x = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂x
            +
            surface_intersect.shading.∂n∂v * surface_intersect.∂v∂x
        )
        ∂n∂y = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂y
            +
            surface_intersect.shading.∂n∂v * surface_intersect.∂v∂y
        )
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        rx_direction = wi - ∂wo∂x + 2f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
        ry_direction = wi - ∂wo∂y + 2f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
        rd = RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
    end
    return f * li(i, rd, scene, depth + 1) * abs(wi ⋅ ns) / pdf
end

function specular_transmit(
        bsdf, i::S, ray::RayDifferentials,
        surface_intersect::SurfaceInteraction, scene::Scene, depth::Int64,
    ) where S<:SamplerIntegrator

    # Compute specular reflection direction `wi` and BSDF value.
    wo = surface_intersect.core.wo
    type = BSDF_TRANSMISSION | BSDF_SPECULAR
    wi, f, pdf, sampled_type = sample_f(
        bsdf, wo, get_2d(i.sampler), type,
    )

    ns = surface_intersect.shading.n
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        return RGBSpectrum(0f0)
    end
    # TODO shift in ray direction instead of normal?
    rd = RayDifferentials(spawn_ray(surface_intersect, wi))
    if ray.has_differentials
        rx_origin = surface_intersect.core.p + surface_intersect.∂p∂x
        ry_origin = surface_intersect.core.p + surface_intersect.∂p∂y
        # Compute differential transmitted directions.
        ∂n∂x = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂x
            +
            surface_intersect.shading.∂n∂v * surface_intersect.∂v∂x
        )
        ∂n∂y = (
            surface_intersect.shading.∂n∂u * surface_intersect.∂u∂y
            +
            surface_intersect.shading.∂n∂v * surface_intersect.∂v∂y
        )
        # The BSDF stores the IOR of the interior of the object being
        # intersected. Compute the relative IOR by first out by assuming
        # that the ray is entering the object.
        η = 1f0 / bsdf.η
        if (ns ⋅ ns) < 0
            # If the ray isn't entering the object, then we need to invert
            # the relative IOR and negate the normal and its derivatives.
            η = 1f0 / η
            ∂n∂x, ∂n∂y, ns = -∂n∂x, -∂n∂y, -ns
        end
        ∂wo∂x = -ray.rx_direction - wo
        ∂wo∂y = -ray.ry_direction - wo
        ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
        ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
        μ = η * (wo ⋅ ns) - abs(wi ⋅ ns)
        ν = η - (η^2 * (wo ⋅ ns)) / abs(wi ⋅ ns)
        ∂μ∂x = ν * ∂dn∂x
        ∂μ∂y = ν * ∂dn∂y
        rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
        ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
        rd = RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
    end
    f * li(i, rd, scene, depth + 1) * abs(wi ⋅ ns) / pdf
end
