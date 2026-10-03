#[derive(Clone, Debug, Default)]
struct Stage {
    first: [f64; 2],
    last: [f64; 2],
    history: [[f64; 2]; 5],
    history_count: usize,
    pushes: u64,
}

impl Stage {
    fn append(&mut self, value: [f64; 2]) {
        if self.history_count < 5 {
            self.history[self.history_count] = value;
            self.history_count += 1;
            return;
        }
        for index in 0..4 {
            self.history[index] = self.history[index + 1];
        }
        self.history[4] = value;
    }

    fn centered_left_box5(&self) -> [f64; 2] {
        let mut output = [0.0f64; 2];
        for axis in 0..2 {
            let mut sum = 0.0;
            if self.pushes == 3 {
                // Keep the same left-to-right multiply/add order as
                // abcurves.smoothing.smooth_dxdy. This is intentional:
                // grouping the repeated edge value as 3*x/5 is not bit exact.
                sum += self.history[0][axis] * 0.2;
                sum += self.history[0][axis] * 0.2;
                sum += self.history[0][axis] * 0.2;
                sum += self.history[1][axis] * 0.2;
                sum += self.history[2][axis] * 0.2;
            } else if self.pushes == 4 {
                sum += self.history[0][axis] * 0.2;
                sum += self.history[0][axis] * 0.2;
                sum += self.history[1][axis] * 0.2;
                sum += self.history[2][axis] * 0.2;
                sum += self.history[3][axis] * 0.2;
            } else {
                sum += self.history[0][axis] * 0.2;
                sum += self.history[1][axis] * 0.2;
                sum += self.history[2][axis] * 0.2;
                sum += self.history[3][axis] * 0.2;
                sum += self.history[4][axis] * 0.2;
            }
            output[axis] = sum;
        }
        output
    }
}

/// Streaming transcription of
/// `delta -> cumulative path -> centered boxcar-5 -> centered boxcar-5 -> path difference`
/// with edge padding at both ends.
#[derive(Clone, Debug, Default)]
pub struct W5Stream {
    cumulative: [f64; 2],
    path: Stage,
    first_stage: Stage,
    previous_smoothed_path: [f64; 2],
    have_previous_smoothed_path: bool,
    flushed: bool,
    real_ticks: u64,
    emitted_ticks: u64,
}

impl W5Stream {
    pub fn new() -> Self {
        Self::default()
    }

    fn push_first_stage(&mut self, value: [f64; 2]) -> Option<[f64; 2]> {
        if self.first_stage.pushes == 0 {
            self.first_stage.first = value;
        }
        self.first_stage.last = value;
        self.first_stage.append(value);
        self.first_stage.pushes += 1;
        if self.first_stage.pushes < 3 {
            return None;
        }
        let smoothed = self.first_stage.centered_left_box5();
        let mut finalized = [0.0f64; 2];
        for axis in 0..2 {
            let previous = if self.have_previous_smoothed_path {
                self.previous_smoothed_path[axis]
            } else {
                0.0
            };
            finalized[axis] = f64::from((smoothed[axis] - previous) as f32);
            self.previous_smoothed_path[axis] = smoothed[axis];
        }
        self.have_previous_smoothed_path = true;
        self.emitted_ticks += 1;
        Some(finalized)
    }

    fn push_path(&mut self, value: [f64; 2]) -> Option<[f64; 2]> {
        if self.path.pushes == 0 {
            self.path.first = value;
        }
        self.path.last = value;
        self.path.append(value);
        self.path.pushes += 1;
        if self.path.pushes < 3 {
            return None;
        }
        let staged = self.path.centered_left_box5();
        self.push_first_stage(staged)
    }

    pub fn push_delta(&mut self, dx: f64, dy: f64) -> Option<[f64; 2]> {
        if self.flushed {
            return None;
        }
        self.cumulative[0] += dx;
        self.cumulative[1] += dy;
        let path = self.cumulative;
        self.real_ticks += 1;
        self.push_path(path)
    }

    pub fn flush(&mut self) -> Vec<[f64; 2]> {
        let mut tail = Vec::new();
        if self.flushed {
            return tail;
        }
        self.flushed = true;
        if self.real_ticks == 0 {
            return tail;
        }
        // Finish the two unavailable right-edge path positions.
        for _ in 0..2 {
            let last = self.path.last;
            if let Some(value) = self.push_path(last) {
                tail.push(value);
            }
        }
        // Then edge-pad the already-complete first stage, exactly as offline.
        for _ in 0..2 {
            let last = self.first_stage.last;
            if let Some(value) = self.push_first_stage(last) {
                tail.push(value);
            }
        }
        tail
    }

    pub fn balanced(&self) -> bool {
        self.emitted_ticks == self.real_ticks
    }
}
