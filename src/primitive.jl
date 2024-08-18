struct GeometricPrimitive{T<:AbstractShape,M<:UberMaterial} <: Primitive
    shape::T
    material::M

    function GeometricPrimitive(
        shape::T, material::UberMaterial=NoMaterial(),
    ) where {T<:AbstractShape}
        new{T, typeof(material)}(shape, material)
    end
end

increase_hit(ray::Ray, t_hit) = Ray(ray; t_max=t_hit)
increase_hit(ray::RayDifferentials, t_hit) = RayDifferentials(ray; t_max=t_hit)

function intersect_p!(
        p::GeometricPrimitive{T}, ray::R,
    )::Tuple{Bool, R, SurfaceInteraction} where {T<:AbstractShape, R<:AbstractRay}
    shape = p.shape
    intersects, t_hit, interaction = intersect(shape, ray)
    !intersects && return false, ray, interaction
    ray = increase_hit(ray, t_hit)
    return true, ray, interaction
end

@noinline function intersect_p(
        p::GeometricPrimitive, ray::Union{Ray,RayDifferentials},
    )
    intersect_p(p.shape, ray)
end
@inline world_bound(p::GeometricPrimitive) = world_bound(p.shape)

function compute_scattering!(
        p::GeometricPrimitive, si::SurfaceInteraction,
        allow_multiple_lobes::Bool, transport::UInt8,
    )
    @real_assert (si.core.n ⋅ si.shading.n) ≥ 0f0
    if p.material.type !== NO_MATERIAL
        return p.material(si, allow_multiple_lobes, transport)
    end
    return BSDF()
end
