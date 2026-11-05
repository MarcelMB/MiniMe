//! Step timings for the speed work (`profile` feature): `span("name")` adds the time until
//! it is dropped to a per-thread table that `take()` returns and clears. Without the
//! feature every call is a no-op.

#[cfg(feature = "profile")]
mod on {
    use std::cell::RefCell;

    #[cfg(not(target_arch = "wasm32"))]
    pub fn now_ms() -> f64 {
        use std::sync::OnceLock;
        use std::time::Instant;
        static T0: OnceLock<Instant> = OnceLock::new();
        T0.get_or_init(Instant::now).elapsed().as_secs_f64() * 1e3
    }

    #[cfg(target_arch = "wasm32")]
    pub fn now_ms() -> f64 {
        use wasm_bindgen::prelude::*;
        #[wasm_bindgen]
        extern "C" {
            #[wasm_bindgen(js_namespace = performance, js_name = now)]
            fn perf_now() -> f64;
        }
        perf_now()
    }

    thread_local! {
        static TABLE: RefCell<Vec<(&'static str, f64)>> = const { RefCell::new(Vec::new()) };
    }

    pub struct Span(&'static str, f64);

    pub fn span(name: &'static str) -> Span {
        Span(name, now_ms())
    }

    impl Drop for Span {
        fn drop(&mut self) {
            let dt = now_ms() - self.1;
            TABLE.with(|t| {
                let mut t = t.borrow_mut();
                match t.iter_mut().find(|(n, _)| *n == self.0) {
                    Some(e) => e.1 += dt,
                    None => t.push((self.0, dt)),
                }
            });
        }
    }

    /// The accumulated `(name, ms)` pairs in first-use order; clears the table.
    pub fn take() -> Vec<(&'static str, f64)> {
        TABLE.with(|t| std::mem::take(&mut *t.borrow_mut()))
    }
}

#[cfg(feature = "profile")]
pub use on::{now_ms, span, take, Span};

#[cfg(not(feature = "profile"))]
mod off {
    pub struct Span;
    #[inline(always)]
    pub fn span(_name: &'static str) -> Span {
        Span
    }
    pub fn take() -> Vec<(&'static str, f64)> {
        Vec::new()
    }
}

#[cfg(not(feature = "profile"))]
pub use off::{span, take, Span};
