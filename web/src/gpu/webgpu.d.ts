// The part of WebGPU the GPU path uses (TypeScript 5.6's DOM lib has none of it; `@webgpu/types` would
// be a dependency for these few lines).

interface GPUAdapterInfo {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly isFallbackAdapter?: boolean;
}

interface GPUSupportedLimits {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupsPerDimension: number;
}

interface GPUAdapter {
  readonly info: GPUAdapterInfo;
  readonly limits: GPUSupportedLimits;
  readonly isFallbackAdapter?: boolean;
  requestDevice(desc?: { requiredLimits?: Record<string, number>; label?: string }): Promise<GPUDevice>;
}

interface GPUDeviceLostInfo {
  readonly reason: string;
  readonly message: string;
}

interface GPUError {
  readonly message: string;
}

interface GPUUncapturedErrorEvent extends Event {
  readonly error: GPUError;
}

interface GPUBuffer {
  readonly size: number;
  mapAsync(mode: number, offset?: number, size?: number): Promise<void>;
  getMappedRange(offset?: number, size?: number): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}

interface GPUBindGroupLayout {
  readonly __brand: "GPUBindGroupLayout";
}

interface GPUBindGroup {
  readonly __brand: "GPUBindGroup";
}

interface GPUShaderModule {
  readonly __brand: "GPUShaderModule";
}

interface GPUComputePipeline {
  getBindGroupLayout(index: number): GPUBindGroupLayout;
}

interface GPUComputePassEncoder {
  setPipeline(p: GPUComputePipeline): void;
  setBindGroup(index: number, g: GPUBindGroup): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}

interface GPUCommandBuffer {
  readonly __brand: "GPUCommandBuffer";
}

interface GPUCommandEncoder {
  beginComputePass(): GPUComputePassEncoder;
  copyBufferToBuffer(src: GPUBuffer, srcOffset: number, dst: GPUBuffer, dstOffset: number, size: number): void;
  finish(): GPUCommandBuffer;
}

interface GPUQueue {
  submit(buffers: GPUCommandBuffer[]): void;
  writeBuffer(buffer: GPUBuffer, offset: number, data: BufferSource, dataOffset?: number, size?: number): void;
  onSubmittedWorkDone(): Promise<void>;
}

interface GPUDevice extends EventTarget {
  readonly queue: GPUQueue;
  readonly lost: Promise<GPUDeviceLostInfo>;
  readonly limits: GPUSupportedLimits;
  createBuffer(desc: { size: number; usage: number; label?: string }): GPUBuffer;
  createShaderModule(desc: { code: string; label?: string }): GPUShaderModule;
  createComputePipelineAsync(desc: {
    layout: "auto";
    compute: { module: GPUShaderModule; entryPoint: string };
    label?: string;
  }): Promise<GPUComputePipeline>;
  createBindGroup(desc: {
    layout: GPUBindGroupLayout;
    entries: { binding: number; resource: { buffer: GPUBuffer } }[];
  }): GPUBindGroup;
  createCommandEncoder(): GPUCommandEncoder;
  destroy(): void;
}

interface GPU {
  requestAdapter(opts?: { powerPreference?: "high-performance" | "low-power" }): Promise<GPUAdapter | null>;
}

interface Navigator {
  readonly gpu?: GPU;
}

interface WorkerNavigator {
  readonly gpu?: GPU;
}

declare const GPUBufferUsage: {
  readonly MAP_READ: number;
  readonly MAP_WRITE: number;
  readonly COPY_SRC: number;
  readonly COPY_DST: number;
  readonly UNIFORM: number;
  readonly STORAGE: number;
};

declare const GPUMapMode: {
  readonly READ: number;
  readonly WRITE: number;
};
