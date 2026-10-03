pub const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug)]
pub struct ProDMPConfig {
    pub n_basis: usize,
    pub alpha: f64,
    pub alpha_phase: f64,
    pub basis_width_scale: f64,
    pub grid_points: usize,
    pub ridge: f64,
}

impl Default for ProDMPConfig {
    fn default() -> Self {
        Self {
            n_basis: 20,
            alpha: 25.0,
            alpha_phase: 3.0,
            basis_width_scale: 1.0,
            grid_points: 2000,
            ridge: 1e-3,
        }
    }
}

/// Pairwise reduction with numpy's eight-lane unrolled block.
pub fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        return values.iter().sum();
    }
    if n <= BLOCK {
        let mut lanes = [
            values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        ];
        let mut index = 8;
        while index < n - (n % 8) {
            for lane in 0..8 {
                lanes[lane] += values[index + lane];
            }
            index += 8;
        }
        let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        while index < n {
            total += values[index];
            index += 1;
        }
        return total;
    }
    let half = (n / 2) - ((n / 2) % 8);
    pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
}

fn interp(x: f64, grid: &[f64], values: &[f64]) -> f64 {
    let last = grid.len() - 1;
    if x < grid[0] {
        return values[0];
    }
    if x > grid[last] {
        return values[last];
    }
    if x == grid[last] {
        return values[last];
    }
    let mut low = 0usize;
    let mut high = last;
    while high - low > 1 {
        let middle = (low + high) / 2;
        if grid[middle] <= x {
            low = middle;
        } else {
            high = middle;
        }
    }
    let slope = (values[low + 1] - values[low]) / (grid[low + 1] - grid[low]);
    slope * (x - grid[low]) + values[low]
}

pub struct ProDMP {
    pub config: ProDMPConfig,
    pub n_basis: usize,
    pub alpha: f64,
    pub alpha_phase: f64,
    #[allow(dead_code)]
    half_alpha: f64,
    s_grid: Vec<f64>,
    phi_grid: Vec<f64>,
    dphi_grid: Vec<f64>,
    phi_columns: Vec<Vec<f64>>,
}

#[derive(Clone)]
pub struct Components {
    pub xi1: Vec<f64>,
    pub xi2: Vec<f64>,
    pub h: Vec<f64>,
}

impl ProDMP {
    pub fn new(config: ProDMPConfig) -> Self {
        let n_basis = config.n_basis;
        let half_alpha = config.alpha / 2.0;
        let points = config.grid_points + 1;
        let step = 1.0 / config.grid_points as f64;
        let mut s_grid: Vec<f64> = (0..points).map(|index| index as f64 * step).collect();
        s_grid[points - 1] = 1.0;

        let phase: Vec<f64> = s_grid
            .iter()
            .map(|value| (-config.alpha_phase * value).exp())
            .collect();
        let centers: Vec<f64> = (0..n_basis)
            .map(|index| (-config.alpha_phase * (index as f64 / (n_basis - 1) as f64)).exp())
            .collect();
        let mut widths = vec![0.0f64; n_basis];
        for index in 0..n_basis - 1 {
            let spacing = centers[index + 1] - centers[index];
            widths[index] = 1.0 / (spacing * spacing).max(EPS);
        }
        widths[n_basis - 1] = widths[n_basis - 2];
        for width in widths.iter_mut() {
            *width *= config.basis_width_scale;
        }

        let n_weights = n_basis + 1;
        let mut forcing = vec![0.0f64; points * n_basis];
        let mut row = vec![0.0f64; n_basis];
        for grid in 0..points {
            for index in 0..n_basis {
                let offset = phase[grid] - centers[index];
                row[index] = (-widths[index] * offset * offset).exp();
            }
            let mut total = pairwise_sum(&row);
            if total < EPS {
                total = EPS;
            }
            for index in 0..n_basis {
                forcing[grid * n_basis + index] = phase[grid] * (row[index] / total);
            }
        }

        let ea: Vec<f64> = s_grid.iter().map(|value| (half_alpha * value).exp()).collect();
        let em: Vec<f64> = s_grid.iter().map(|value| (-half_alpha * value).exp()).collect();
        let mut p1 = vec![0.0f64; points * n_basis];
        let mut p2 = vec![0.0f64; points * n_basis];
        for grid in 1..points {
            let dx = s_grid[grid] - s_grid[grid - 1];
            for index in 0..n_basis {
                let previous = (grid - 1) * n_basis + index;
                let current = grid * n_basis + index;
                let integ1_previous = s_grid[grid - 1] * ea[grid - 1] * forcing[previous];
                let integ1_current = s_grid[grid] * ea[grid] * forcing[current];
                let integ2_previous = ea[grid - 1] * forcing[previous];
                let integ2_current = ea[grid] * forcing[current];
                p1[current] = p1[previous] + 0.5 * (integ1_current + integ1_previous) * dx;
                p2[current] = p2[previous] + 0.5 * (integ2_current + integ2_previous) * dx;
            }
        }

        let mut phi_grid = vec![0.0f64; points * n_weights];
        let mut dphi_grid = vec![0.0f64; points * n_weights];
        for grid in 0..points {
            for index in 0..n_basis {
                let at = grid * n_basis + index;
                phi_grid[grid * n_weights + index] =
                    (s_grid[grid] * em[grid]) * p2[at] - em[grid] * p1[at];
                dphi_grid[grid * n_weights + index] = em[grid]
                    * (((1.0 - half_alpha * s_grid[grid]) * p2[at]) + half_alpha * p1[at]);
            }
            phi_grid[grid * n_weights + n_basis] =
                1.0 - em[grid] * (1.0 + half_alpha * s_grid[grid]);
            dphi_grid[grid * n_weights + n_basis] =
                half_alpha * half_alpha * s_grid[grid] * em[grid];
        }

        let phi_columns = (0..n_weights)
            .map(|index| {
                (0..points)
                    .map(|grid| phi_grid[grid * n_weights + index])
                    .collect()
            })
            .collect();

        Self {
            config,
            n_basis,
            alpha: config.alpha,
            alpha_phase: config.alpha_phase,
            half_alpha,
            s_grid,
            phi_grid,
            dphi_grid,
            phi_columns,
        }
    }

    pub fn n_weights(&self) -> usize {
        self.n_basis + 1
    }

    pub fn s_grid_value(&self, index: usize) -> f64 {
        self.s_grid[index]
    }

    pub fn phi_grid(&self) -> &[f64] {
        &self.phi_grid
    }

    pub fn dphi_grid(&self) -> &[f64] {
        &self.dphi_grid
    }

    /// Components for positive integer duration, t=1..d, tau=d, B=0.
    pub fn canonical_components(&self, duration: usize) -> Components {
        let weights = self.n_weights();
        let mut xi1 = vec![0.0f64; duration];
        let mut xi2 = vec![0.0f64; duration];
        let mut h = vec![0.0f64; duration * weights];
        let a = self.alpha / (2.0 * duration as f64);
        for step in 0..duration {
            let t = (step + 1) as f64;
            let e = (-a * t).exp();
            xi2[step] = t * e;
            xi1[step] = e - (-a) * xi2[step];
            let s = t / duration as f64;
            for index in 0..weights {
                h[step * weights + index] = interp(s, &self.s_grid, &self.phi_columns[index]);
            }
        }
        Components { xi1, xi2, h }
    }

    pub fn generate_deltas(
        &self,
        components: &Components,
        weights: &[f64],
        velocity: [f64; 2],
        out: &mut [[f64; 2]],
    ) {
        let n_weights = self.n_weights();
        let duration = out.len();
        let mut previous = [0.0f64; 2];
        for step in 0..duration {
            let mut position = [0.0f64; 2];
            for axis in 0..2 {
                let mut total = 0.0f64;
                for index in 0..n_weights {
                    total += components.h[step * n_weights + index] * weights[index * 2 + axis];
                }
                position[axis] = components.xi2[step] * velocity[axis] + total;
            }
            out[step] = if step == 0 {
                position
            } else {
                [position[0] - previous[0], position[1] - previous[1]]
            };
            previous = position;
        }
    }
}

pub struct ComponentCache {
    prodmp: ProDMP,
    cache: Vec<Option<Components>>,
}

impl ComponentCache {
    pub fn new(config: ProDMPConfig, horizon: usize) -> Self {
        Self {
            prodmp: ProDMP::new(config),
            cache: (0..=horizon).map(|_| None).collect(),
        }
    }

    pub fn prodmp(&self) -> &ProDMP {
        &self.prodmp
    }

    pub fn prewarm(&mut self) {
        for duration in 1..self.cache.len() {
            self.components(duration);
        }
    }

    pub fn components(&mut self, duration: usize) -> &Components {
        if self.cache[duration].is_none() {
            self.cache[duration] = Some(self.prodmp.canonical_components(duration));
        }
        self.cache[duration].as_ref().unwrap()
    }
}
