struct ProjectiveCamera <: Camera
    core::CameraCore
    camera_to_screen::Transformation
    raster_to_camera::Transformation
    screen_to_raster::Transformation
    raster_to_screen::Transformation

    lens_radius::Float32
    focal_distance::Float32

    function ProjectiveCamera(
            camera_to_world::Transformation, camera_to_screen::Transformation,
            screen_window::Bounds2,
            shutter_open::Float32, shutter_close::Float32,
            lens_radius::Float32, focal_distance::Float32,
            film::Film,
        )
        core = CameraCore(camera_to_world, shutter_open, shutter_close)
        # Computer projective camera transformations.
        resolution = scale(film.resolution..., 1)
        window_width = screen_window.p_max .- screen_window.p_min
        inv_bounds = scale((1f0 ./ window_width)..., 1)

        offset = translate(Vec3f(
            (-screen_window.p_min)..., 0f0,
        ))

        screen_to_raster = resolution * inv_bounds * offset
        raster_to_screen = inv(offset) * inv(inv_bounds) * inv(resolution)
        raster_to_camera = inv(camera_to_screen) * raster_to_screen

        new(
            core,
            camera_to_screen, raster_to_camera,
            screen_to_raster, raster_to_screen,
            lens_radius, focal_distance,
        )
    end
end

struct PerspectiveCamera <: Camera
    core::ProjectiveCamera
    """
    Precomputed change of rays as we shift pixels on the plane in x-direction.
    """
    dx_camera::Vec3f
    """
    Precomputed change of rays as we shift pixels on the plane in y-direction.
    """
    dy_camera::Vec3f
    A::Float32

    """
    - `screen_window::Bounds2`: Screen space extent of the image.
    """
    function PerspectiveCamera(
        camera_to_world::Transformation, screen_window::Bounds2,
        shutter_open::Float32, shutter_close::Float32,
        lens_radius::Float32, focal_distance::Float32,
        fov::Float32, film::Film,
    )
        pc = ProjectiveCamera(
            inv(camera_to_world),
            perspective(fov, 0.01f0, 1000.0f0),
            screen_window, shutter_open, shutter_close,
            lens_radius, focal_distance
        )

        p_min = pc.raster_to_camera(Point3f(0))
        p_max = pc.raster_to_camera(Point3f(
            film.resolution[1], film.resolution[2], 0f0,
        ))
        dx_camera = pc.raster_to_camera(Point3f(1, 0, 0)) - p_min
        dy_camera = pc.raster_to_camera(Point3f(0, 1, 0)) - p_min
        p = (p_min[1:2] ./ p_min[3]) - (p_max[1:2] ./ p_max[3])
        A = abs(p[1] * p[2])

        new(pc, dx_camera, dy_camera, A)
    end
end

@inline get_film(c::PerspectiveCamera)::Film = c.core.core.film

@inline function generate_ray(
        camera::PerspectiveCamera, sample::CameraSample,
    )::Tuple{Ray,Float32}
    # Compute raster & camera sample positions.
    # p_film -> in pixels
    p_pixel = Point3f(sample.film..., 0f0)
    p_camera = camera.core.raster_to_camera(p_pixel)
    o = Point3f(0)
    d = normalize(Vec3f(p_camera))
    # Modify ray for depth of field.
    if camera.core.lens_radius > 0
        # Sample points on lens.
        p_lens = camera.core.lens_radius * concentric_sample_disk(sample.lens)
        # Compute point on plane of focus.
        t = camera.core.focal_distance / d[3]
        p_focus = o .+ d * t
        # Update ray for effects of lens.
        o = Point3f(p_lens[1], p_lens[2], 0f0)
        d = normalize(Vec3f(p_focus .- o))
    end

    time = lerp(
        camera.core.core.shutter_open,
        camera.core.core.shutter_close,
        sample.time,
    )
    # TODO add medium
    ctw = camera.core.core.camera_to_world
    o = ctw(o)
    d = ctw(d)
    return Ray(d=normalize(d), o=o, time=time), 1.0f0
end
