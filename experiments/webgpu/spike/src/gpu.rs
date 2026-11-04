//! The preprocessing chain (median 7x7, shift, tophat 15) on the GPU through `wgpu`: the same
//! WGSL as `experiments/webgpu/kernels.js`, the same pass structure as `gpuprep.js`.

const UNPACK: &str = include_str!("kernels/unpack.wgsl");
const TOPHAT_FINAL: &str = include_str!("kernels/tophat_final.wgsl");
const MEDIAN: &str = include_str!("kernels/median.wgsl");
const SHIFT: &str = include_str!("kernels/shift.wgsl");
const ERODE_ROWS: &str = include_str!("kernels/erode_rows.wgsl");
const ERODE_COLS: &str = include_str!("kernels/erode_cols.wgsl");
const DILATE_ROWS: &str = include_str!("kernels/dilate_rows.wgsl");
const DILATE_COLS: &str = include_str!("kernels/dilate_cols.wgsl");

fn pipeline(device: &wgpu::Device, label: &str, src: &str) -> wgpu::ComputePipeline {
    let desc = wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    };
    // native only: UNCHECKED=1 drops wgpu's robustness bounds checks (the browser always keeps its own)
    #[cfg(not(target_arch = "wasm32"))]
    let module = if std::env::var_os("UNCHECKED").is_some() {
        unsafe { device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
    } else {
        device.create_shader_module(desc)
    };
    #[cfg(target_arch = "wasm32")]
    let module = device.create_shader_module(desc);
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn ceil(a: usize, b: usize) -> u32 {
    a.div_ceil(b) as u32
}

/// Adapter and device with the larger buffer limits the batches need.
pub async fn init_device() -> Result<(wgpu::Device, wgpu::Queue, String), String> {
    #[cfg(target_arch = "wasm32")]
    let backends = wgpu::Backends::BROWSER_WEBGPU;
    #[cfg(not(target_arch = "wasm32"))]
    let backends = wgpu::Backends::PRIMARY;
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    let instance = wgpu::Instance::new(desc);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("no adapter: {e}"))?;
    let ai = adapter.get_info();
    let info = format!("{} / {:?} / {}", ai.name, ai.backend, ai.driver);
    let lim = adapter.limits();
    let mut limits = wgpu::Limits::default();
    limits.max_buffer_size = lim.max_buffer_size.min((1 << 31) - 4);
    limits.max_storage_buffer_binding_size = lim.max_storage_buffer_binding_size.min((1 << 31) - 4);
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_limits: limits,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("no device: {e}"))?;
    Ok((device, queue, info))
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: String,
    pub h: usize,
    pub w: usize,
    pub max_b: usize,
    unpack: wgpu::ComputePipeline,
    median: wgpu::ComputePipeline,
    shift: wgpu::ComputePipeline,
    erode_rows: wgpu::ComputePipeline,
    erode_cols: wgpu::ComputePipeline,
    dilate_rows: wgpu::ComputePipeline,
    dilate_cols: wgpu::ComputePipeline,
    tophat_final: wgpu::ComputePipeline,
    scratch: [wgpu::Buffer; 3],
    in_packed: wgpu::Buffer,
    out_packed: wgpu::Buffer,
    shifts: wgpu::Buffer,
    read: wgpu::Buffer,
    params: wgpu::Buffer,
}

impl Gpu {
    pub async fn new(h: usize, w: usize, max_b: usize) -> Result<Gpu, String> {
        let (device, queue, info) = init_device().await?;
        let hw = h * w;
        let words = (max_b * hw).div_ceil(4);
        let buf = |size: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let s = wgpu::BufferUsages::STORAGE;
        let big = || buf(words * 16, s);
        Ok(Gpu {
            unpack: pipeline(&device, "unpack", UNPACK),
            median: pipeline(&device, "median", MEDIAN),
            shift: pipeline(&device, "shift", SHIFT),
            erode_rows: pipeline(&device, "erode_rows", ERODE_ROWS),
            erode_cols: pipeline(&device, "erode_cols", ERODE_COLS),
            dilate_rows: pipeline(&device, "dilate_rows", DILATE_ROWS),
            dilate_cols: pipeline(&device, "dilate_cols", DILATE_COLS),
            tophat_final: pipeline(&device, "tophat_final", TOPHAT_FINAL),
            scratch: [big(), big(), big()],
            in_packed: buf(words * 4, s | wgpu::BufferUsages::COPY_DST),
            out_packed: buf(words * 4, s | wgpu::BufferUsages::COPY_SRC),
            shifts: buf(max_b * 8, s | wgpu::BufferUsages::COPY_DST),
            read: buf(
                words * 4,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            params: buf(16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            device,
            queue,
            info,
            h,
            w,
            max_b,
        })
    }

    fn pass(
        &self,
        enc: &mut wgpu::CommandEncoder,
        pipe: &wgpu::ComputePipeline,
        bufs: &[&wgpu::Buffer],
        grid: (u32, u32, u32),
    ) {
        let layout = pipe.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = std::iter::once(&self.params)
            .chain(bufs.iter().copied())
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut cp = enc.begin_compute_pass(&Default::default());
        cp.set_pipeline(pipe);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(grid.0, grid.1, grid.2);
    }

    /// Upload `frames` / `shifts` of `n` frames, run median, shift, tophat, copy the result to the readback buffer.
    pub fn submit(&self, frames: &[u8], shifts: &[f32], n: usize) {
        let (h, w) = (self.h, self.w);
        let total = n * h * w;
        let whole = total & !3;
        let q = &self.queue;
        let params: [u32; 4] = [h as u32, w as u32, n as u32, total as u32];
        q.write_buffer(
            &self.params,
            0,
            &params.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>(),
        );
        q.write_buffer(&self.in_packed, 0, &frames[..whole]);
        if total > whole {
            let mut tail = [0u8; 4];
            tail[..total - whole].copy_from_slice(&frames[whole..total]);
            q.write_buffer(&self.in_packed, whole as u64, &tail);
        }
        let sh: Vec<u8> = shifts[..2 * n].iter().flat_map(|v| v.to_le_bytes()).collect();
        q.write_buffer(&self.shifts, 0, &sh);

        let words = total.div_ceil(4);
        let groups = words.div_ceil(256);
        let g1 = (groups.min(65535) as u32, groups.div_ceil(65535) as u32, 1);
        let [a, b, c] = &self.scratch;
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.pass(&mut enc, &self.unpack, &[&self.in_packed, a], g1);
        self.pass(
            &mut enc,
            &self.median,
            &[a, b],
            (ceil(w, 16), ceil(h, 16), n as u32),
        );
        self.pass(
            &mut enc,
            &self.shift,
            &[b, c, &self.shifts],
            (ceil(w, 16), ceil(h, 16), n as u32),
        );
        let g = (ceil(w, 32), ceil(h, 8), n as u32);
        self.pass(&mut enc, &self.erode_rows, &[c, a], g);
        self.pass(&mut enc, &self.erode_cols, &[a, b], g);
        self.pass(&mut enc, &self.dilate_rows, &[b, a], g);
        self.pass(&mut enc, &self.dilate_cols, &[a, b], g);
        self.pass(&mut enc, &self.tophat_final, &[c, b, &self.out_packed], g1);
        enc.copy_buffer_to_buffer(&self.out_packed, 0, &self.read, 0, (words * 4) as u64);
        q.submit([enc.finish()]);
    }

    /// Wait for the batch queued by `submit` and copy its `n * h * w` result bytes into `out`.
    pub async fn read(&self, n: usize, out: &mut [u8]) -> Result<(), String> {
        let bytes = n * self.h * self.w;
        let slice = self.read.slice(..(bytes.div_ceil(4) * 4) as u64);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.await
            .map_err(|_| "map cancelled".to_string())?
            .map_err(|e| e.to_string())?;
        {
            let view = slice.get_mapped_range().map_err(|e| e.to_string())?;
            out.copy_from_slice(&view[..bytes]);
        }
        self.read.unmap();
        Ok(())
    }

    /// Upload, compute and read one batch.
    pub async fn run(&self, frames: &[u8], shifts: &[f32], n: usize, out: &mut [u8]) -> Result<(), String> {
        self.submit(frames, shifts, n);
        self.read(n, out).await
    }
}

// ---------------------------------------------------------------------------
// Pass 2 products on a resident tile
// ---------------------------------------------------------------------------

const GEMM_T: &str = include_str!("kernels/gemm_t.wgsl");
const GEMM_N: &str = include_str!("kernels/gemm_n.wgsl");

/// `Y^T Q` and `Y Z` of one `(T, P)` uint8 tile held on the GPU (mean removed in the kernel).
pub struct GpuProducts {
    device: wgpu::Device,
    queue: wgpu::Queue,
    raw: wgpu::Buffer,
    mean: wgpu::Buffer,
    pipe_t: wgpu::ComputePipeline,
    pipe_n: wgpu::ComputePipeline,
    t: usize,
    p: usize,
}

impl GpuProducts {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        raw: &[u8],
        t: usize,
        p: usize,
        mean: &[f32],
    ) -> Self {
        let buf = |size: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let words = raw.len().div_ceil(4);
        let rb = buf(
            words * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let whole = raw.len() & !3;
        queue.write_buffer(&rb, 0, &raw[..whole]);
        if raw.len() > whole {
            let mut tail = [0u8; 4];
            tail[..raw.len() - whole].copy_from_slice(&raw[whole..]);
            queue.write_buffer(&rb, whole as u64, &tail);
        }
        let mb = buf(p * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        queue.write_buffer(&mb, 0, &f32_bytes(mean));
        GpuProducts {
            pipe_t: pipeline(device, "gemm_t", GEMM_T),
            pipe_n: pipeline(device, "gemm_n", GEMM_N),
            device: device.clone(),
            queue: queue.clone(),
            raw: rb,
            mean: mb,
            t,
            p,
        }
    }

    /// `transposed = false`: `Y^T B` `(T, n)` from `B` `(P, n)`; `true`: `Y B` `(P, n)` from `B` `(T, n)`.
    pub async fn product(&self, transposed: bool, b: &[f32], n: usize) -> Result<Vec<f32>, String> {
        let (m, k, pipe) = if transposed {
            (self.p, self.t, &self.pipe_n)
        } else {
            (self.t, self.p, &self.pipe_t)
        };
        assert_eq!(b.len(), k * n);
        let dev = &self.device;
        let mk = |size: usize, usage: wgpu::BufferUsages| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        let g = mk(16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let bb = mk(
            b.len() * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let out = mk(
            m * n * 4,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let rd = mk(
            m * n * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let params: Vec<u8> = [m as u32, n as u32, k as u32, self.p as u32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        self.queue.write_buffer(&g, 0, &params);
        self.queue.write_buffer(&bb, 0, &f32_bytes(b));
        let layout = pipe.get_bind_group_layout(0);
        let bufs = [&g, &self.raw, &self.mean, &bb, &out];
        let entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            cp.set_pipeline(pipe);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &rd, 0, (m * n * 4) as u64);
        self.queue.submit([enc.finish()]);
        let slice = rd.slice(..);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        #[cfg(not(target_arch = "wasm32"))]
        let _ = dev.poll(wgpu::PollType::wait_indefinitely());
        rx.await
            .map_err(|_| "map cancelled".to_string())?
            .map_err(|e| e.to_string())?;
        let res = {
            let view = slice.get_mapped_range().map_err(|e| e.to_string())?;
            view.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        rd.unmap();
        Ok(res)
    }
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
