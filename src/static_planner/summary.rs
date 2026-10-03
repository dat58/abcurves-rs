pub const SUMMARY_LEN: usize = 62;
const EPS: f64 = 1e-6;

#[inline]
fn norm2_f32(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

#[inline]
fn norm2(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

/// NumPy's contiguous float64 pairwise reduction for lengths below 128.
fn pairwise_sum_f64(values: &[f64]) -> f64 {
    let length = values.len();
    if length < 8 {
        let mut result = -0.0f64;
        for value in values {
            result += value;
        }
        return result;
    }
    let mut lanes = [
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
    ];
    let mut index = 8usize;
    while index + 8 <= length {
        for lane in 0..8 {
            lanes[lane] += values[index + lane];
        }
        index += 8;
    }
    let mut result = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    while index < length {
        result += values[index];
        index += 1;
    }
    result
}

/// Mirror features._slope, which widens its float32 input to float64.
fn slope_f32_values(values: &[f32]) -> f64 {
    let length = values.len();
    if length <= 1 {
        return 0.0;
    }
    let widened: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
    let mean = pairwise_sum_f64(&widened) / length as f64;
    let x_mean = 0.5 * (length - 1) as f64;
    let mut denominators = vec![0.0f64; length];
    let mut numerators = vec![0.0f64; length];
    for index in 0..length {
        let x = index as f64 - x_mean;
        denominators[index] = x * x;
        numerators[index] = x * (widened[index] - mean);
    }
    let denominator = pairwise_sum_f64(&denominators);
    let numerator = pairwise_sum_f64(&numerators);
    if denominator <= 1e-9 {
        return 0.0;
    }
    numerator / denominator
}

/// Mirror features._linear_slope on float64 values.
fn linear_slope(values: &[f64]) -> f64 {
    let length = values.len();
    if length <= 1 {
        return 0.0;
    }
    let mut mean = 0.0f64;
    for value in values {
        mean += value;
    }
    mean /= length as f64;
    let x_mean = 0.5 * (length - 1) as f64;
    let mut denominator = 0.0f64;
    let mut numerator = 0.0f64;
    for index in 0..length {
        let x = index as f64 - x_mean;
        denominator += x * x;
        numerator += x * (values[index] - mean);
    }
    if denominator <= EPS {
        return 0.0;
    }
    numerator / denominator
}

pub fn distance_over_radius(target: [f64; 2], radius: f64) -> f32 {
    // Match summary_features: distance/radius is evaluated before fields are
    // packed into float32 research-row arrays.
    (target[0].hypot(target[1]) / radius.max(1e-6)) as f32
}

pub fn raw_summary62(
    prefix: &[[f32; 2]],
    target_x_f32: f32,
    target_y_f32: f32,
    radius_f32: f32,
    progress_f32: f32,
    distance_over_radius_f32: f32,
) -> [f64; SUMMARY_LEN] {
    let n = prefix.len();
    let mut out = [0.0f64; SUMMARY_LEN];

    // The live builder first writes target geometry into a float32 one-row
    // research record, then the primitive layer reads it back as float64.
    let tx = f64::from(target_x_f32);
    let ty = f64::from(target_y_f32);
    let radius = f64::from(radius_f32).max(EPS);
    let target_norm_unclipped = norm2(tx, ty);
    let target_distance = target_norm_unclipped.max(EPS);
    let (toward_x, toward_y) = if target_norm_unclipped <= EPS {
        (1.0, 0.0)
    } else {
        (tx / target_norm_unclipped, ty / target_norm_unclipped)
    };
    let tangent_x = -toward_y;
    let tangent_y = toward_x;

    let mut speeds = vec![0.0f64; n];
    let mut prefix_x = 0.0f64;
    let mut prefix_y = 0.0f64;
    let mut path_length = 0.0f64;
    let mut peak_speed = 0.0f64;
    for index in 0..n {
        let x = f64::from(prefix[index][0]);
        let y = f64::from(prefix[index][1]);
        prefix_x += x;
        prefix_y += y;
        let speed = norm2(x, y);
        speeds[index] = speed;
        path_length += speed;
        if index == 0 || speed > peak_speed {
            peak_speed = speed;
        }
    }
    let prefix_distance = norm2(prefix_x, prefix_y);
    let mean_speed = if n > 0 { path_length / n as f64 } else { 0.0 };
    let mut speed_variance = 0.0f64;
    for index in 0..n {
        let delta = speeds[index] - mean_speed;
        speed_variance += delta * delta;
    }
    let speed_std = if n > 0 {
        (speed_variance / n as f64).sqrt()
    } else {
        0.0
    };

    let (last_vx, last_vy, speed_at_b32) = if n > 0 {
        (
            f64::from(prefix[n - 1][0]),
            f64::from(prefix[n - 1][1]),
            norm2_f32(prefix[n - 1][0], prefix[n - 1][1]),
        )
    } else {
        (0.0, 0.0, 0.0f32)
    };
    let (last_ax32, last_ay32, accel_at_b32) = if n >= 2 {
        let ax = prefix[n - 1][0] - prefix[n - 2][0];
        let ay = prefix[n - 1][1] - prefix[n - 2][1];
        (ax, ay, norm2_f32(ax, ay))
    } else {
        (0.0f32, 0.0f32, 0.0f32)
    };
    let last_ax = f64::from(last_ax32);
    let last_ay = f64::from(last_ay32);

    let mut inside32 = 0.0f32;
    let mut velocity_toward32 = 0.0f32;
    let mut velocity_lateral32 = 0.0f32;
    let mut recent_zero32 = 0.0f32;
    let mut recent_sign_flip32 = 0.0f32;
    let mut recent_direction_change32 = 0.0f32;
    let mut recent_speed_slope32 = 0.0f32;
    let mut recent_accel_slope32 = 0.0f32;
    let mut duration32 = 0.0f32;
    let mut recent_mean = 0.0f64;
    if n > 0 {
        duration32 = n as f32;
        inside32 = if target_norm_unclipped <= f64::from(radius_f32).max(0.0) {
            1.0
        } else {
            0.0
        };
        // causal_context_arrays uses a tighter 1e-9 target-unit threshold.
        let (causal_tx, causal_ty) = if target_norm_unclipped > 1e-9 {
            (tx / target_norm_unclipped, ty / target_norm_unclipped)
        } else {
            (1.0, 0.0)
        };
        velocity_toward32 = (last_vx * causal_tx + last_vy * causal_ty) as f32;
        velocity_lateral32 = (last_vx * causal_ty - last_vy * causal_tx) as f32;

        let recent_length = n.min(40);
        let start = n - recent_length;
        let mut recent_magnitude = vec![0.0f32; recent_length];
        let mut recent_accel_magnitude = vec![0.0f32; recent_length];
        let mut zeros = 0usize;
        for step in 0..recent_length {
            let index = start + step;
            let magnitude = norm2_f32(prefix[index][0], prefix[index][1]);
            recent_magnitude[step] = magnitude;
            if magnitude <= 0.0 {
                zeros += 1;
            }
            let (ax, ay) = if step == 0 {
                (0.0f32, 0.0f32)
            } else {
                (
                    prefix[index][0] - prefix[index - 1][0],
                    prefix[index][1] - prefix[index - 1][1],
                )
            };
            recent_accel_magnitude[step] = norm2_f32(ax, ay);
        }
        recent_zero32 = (zeros as f64 / recent_length as f64) as f32;
        recent_speed_slope32 = slope_f32_values(&recent_magnitude) as f32;
        recent_accel_slope32 = slope_f32_values(&recent_accel_magnitude) as f32;

        let mut flips = 0usize;
        let mut flip_denominator = 0usize;
        for axis in 0..2 {
            let mut have_previous = false;
            let mut previous_sign = 0i32;
            for index in start..n {
                let value = prefix[index][axis];
                if f64::from(value).abs() > 0.0 {
                    let sign = if value > 0.0 { 1 } else { -1 };
                    if have_previous {
                        flip_denominator += 1;
                        if sign != previous_sign {
                            flips += 1;
                        }
                    }
                    previous_sign = sign;
                    have_previous = true;
                }
            }
        }
        if flip_denominator > 0 {
            recent_sign_flip32 = (flips as f64 / flip_denominator as f64) as f32;
        }

        if recent_length > 2 {
            let mut direction_changes = 0usize;
            let mut direction_denominator = 0usize;
            let mut previous_valid = false;
            let mut previous_ux = 0.0f64;
            let mut previous_uy = 0.0f64;
            for index in start..n {
                // The source expression divides float32 arrays before assigning
                // into its float64 unit array, hence the explicit float32 quotient.
                let magnitude = recent_magnitude[index - start];
                let valid = magnitude > 1e-6;
                if valid {
                    let ux = f64::from(prefix[index][0] / magnitude);
                    let uy = f64::from(prefix[index][1] / magnitude);
                    if previous_valid {
                        direction_denominator += 1;
                        // The reference threshold is this literal, not 1/sqrt(2).
                        #[allow(clippy::approx_constant)]
                        if previous_ux * ux + previous_uy * uy < 0.7071 {
                            direction_changes += 1;
                        }
                    }
                    previous_ux = ux;
                    previous_uy = uy;
                }
                previous_valid = valid;
            }
            if direction_denominator > 0 {
                recent_direction_change32 =
                    (direction_changes as f64 / direction_denominator as f64) as f32;
            }
        }
    }

    // Path relative to B. For an empty prefix the authority deliberately
    // supplies one zero point via _prefix_path_at_b.
    let mut crossed = 0.0f64;
    let mut overshot = 0.0f64;
    let mut min_prefix_distance = target_distance;
    let mut near_count = 0usize;
    let path_points = if n > 0 { n } else { 1 };
    let mut cumulative_x = 0.0f64;
    let mut cumulative_y = 0.0f64;
    let near_threshold = (radius * 2.0).max(radius + 4.0);
    let mut max_axis = f64::NEG_INFINITY;
    for index in 0..path_points {
        let (px, py) = if n > 0 {
            cumulative_x += f64::from(prefix[index][0]);
            cumulative_y += f64::from(prefix[index][1]);
            (cumulative_x - prefix_x, cumulative_y - prefix_y)
        } else {
            (0.0, 0.0)
        };
        let distance = norm2(tx - px, ty - py);
        if index == 0 || distance < min_prefix_distance {
            min_prefix_distance = distance;
        }
        if distance <= radius {
            crossed = 1.0;
        }
        if distance <= near_threshold {
            near_count += 1;
        }
        let axis = px * toward_x + py * toward_y;
        if axis > max_axis {
            max_axis = axis;
        }
    }
    if max_axis > target_distance * 1.03 {
        overshot = 1.0;
    }
    let near_rate = near_count as f64 / path_points as f64;

    let early_length = n.min(20);
    let recent20_length = n.min(20);
    let mut early_mean = 0.0f64;
    for index in 0..early_length {
        early_mean += speeds[index];
    }
    for index in n - recent20_length..n {
        recent_mean += speeds[index];
    }
    if early_length > 0 {
        early_mean /= early_length as f64;
    }
    if recent20_length > 0 {
        recent_mean /= recent20_length as f64;
    }
    let speed_drop_recent = early_mean - recent_mean;

    let movement_norm = norm2(last_vx, last_vy).max(EPS);
    let alignment_cos = (last_vx * toward_x + last_vy * toward_y) / movement_norm;
    let alignment_sin = (last_vx * tangent_x + last_vy * tangent_y) / movement_norm;
    let last_norm = norm2(last_vx, last_vy);
    let direction_error = if last_norm <= EPS && target_norm_unclipped <= EPS {
        0.0
    } else if last_norm <= EPS || target_norm_unclipped <= EPS {
        90.0
    } else {
        let cosine =
            ((last_vx * tx + last_vy * ty) / (last_norm * target_norm_unclipped)).clamp(-1.0, 1.0);
        cosine.acos().to_degrees()
    };

    let mut jerk = 0.0f64;
    if n >= 4 {
        let a0x = f64::from(prefix[n - 2][0]) - f64::from(prefix[n - 3][0]);
        let a0y = f64::from(prefix[n - 2][1]) - f64::from(prefix[n - 3][1]);
        let a1x = f64::from(prefix[n - 1][0]) - f64::from(prefix[n - 2][0]);
        let a1y = f64::from(prefix[n - 1][1]) - f64::from(prefix[n - 2][1]);
        jerk = norm2(a1x - a0x, a1y - a0y);
    }

    let prefix_target_axis = prefix_x * toward_x + prefix_y * toward_y;
    let prefix_lateral = (prefix_x * tangent_x + prefix_y * tangent_y) / radius;
    let accel_toward = last_ax * toward_x + last_ay * toward_y;
    let accel_lateral = last_ax * tangent_x + last_ay * tangent_y;
    let distance_over_radius = f64::from(distance_over_radius_f32);
    let recent_speed_slope = f64::from(recent_speed_slope32);
    let recent_accel_slope = f64::from(recent_accel_slope32);
    let speed_at_b = f64::from(speed_at_b32);
    let accel_at_b = f64::from(accel_at_b32);

    out[0] = f64::from(duration32);
    out[1] = prefix_x;
    out[2] = prefix_y;
    out[3] = prefix_distance;
    out[4] = path_length;
    out[5] = prefix_distance / path_length.max(EPS);
    out[6] = prefix_target_axis;
    out[7] = prefix_lateral;
    out[8] = tx;
    out[9] = ty;
    out[10] = target_distance;
    out[11] = radius;
    out[12] = distance_over_radius;
    out[13] = toward_x;
    out[14] = toward_y;
    out[15] = f64::from(progress_f32);
    out[16] = f64::from(inside32);
    out[17] = if distance_over_radius <= 2.0 || inside32 > 0.5 {
        1.0
    } else {
        0.0
    };
    out[18] = crossed;
    out[19] = overshot;
    out[20] = min_prefix_distance / radius;
    out[21] = near_rate;
    out[22] = last_vx;
    out[23] = last_vy;
    out[24] = last_vx / movement_norm;
    out[25] = last_vy / movement_norm;
    out[26] = speed_at_b;
    out[27] = mean_speed;
    out[28] = peak_speed;
    out[29] = speed_std;
    out[30] = speed_at_b / target_distance;
    out[31] = speed_at_b / radius;
    out[32] = f64::from(velocity_toward32);
    out[33] = f64::from(velocity_lateral32);
    out[34] = alignment_cos;
    out[35] = alignment_sin;
    out[36] = direction_error;
    out[37] = accel_at_b;
    out[38] = accel_toward;
    out[39] = accel_lateral;
    out[40] = recent_speed_slope;
    out[41] = recent_accel_slope;
    out[42] = if recent_speed_slope < -0.03 || accel_at_b < -0.03 {
        1.0
    } else {
        0.0
    };
    out[43] = speed_drop_recent;
    out[44] = jerk;
    out[45] = (speed_at_b - mean_speed).abs();
    out[46] = f64::from(recent_zero32);
    out[47] = f64::from(recent_sign_flip32);
    out[48] = f64::from(recent_direction_change32);
    out[49] = if speed_at_b > 0.5 || recent_mean > 0.5 {
        1.0
    } else {
        0.0
    };
    out[50] = if distance_over_radius <= 1.5 && speed_at_b < 1.0f64.max(mean_speed * 0.5) {
        1.0
    } else {
        0.0
    };

    // Prefix-shape layer: recent curvature, approach, four speed-profile bins,
    // and target-frame lateral drift.
    let shape_start = n.saturating_sub(16);
    let mut curvature_signed_sum = 0.0f64;
    let mut curvature_absolute_sum = 0.0f64;
    let mut turn_angles = Vec::with_capacity(15);
    if n - shape_start >= 2 {
        for index in shape_start..n - 1 {
            let x0 = f64::from(prefix[index][0]);
            let y0 = f64::from(prefix[index][1]);
            let x1 = f64::from(prefix[index + 1][0]);
            let y1 = f64::from(prefix[index + 1][1]);
            let n0 = norm2(x0, y0);
            let n1 = norm2(x1, y1);
            if n0 > 1e-6 && n1 > 1e-6 {
                let denominator = (n0 * n1).max(1e-9);
                let sine = (x0 * y1 - y0 * x1) / denominator;
                curvature_signed_sum += sine;
                curvature_absolute_sum += sine.abs();
                let dot = ((x0 * x1 + y0 * y1) / denominator).clamp(-1.0, 1.0);
                turn_angles.push(dot.acos().to_degrees());
            }
        }
    }
    let turn_count = turn_angles.len();
    let (curvature_signed, curvature_absolute, direction_change_mean) = if turn_count > 0 {
        let mut mean = 0.0f64;
        for angle in &turn_angles {
            mean += angle;
        }
        (
            curvature_signed_sum / turn_count as f64,
            curvature_absolute_sum / turn_count as f64,
            mean / turn_count as f64,
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    let direction_change_slope = if turn_count >= 2 {
        linear_slope(&turn_angles)
    } else {
        0.0
    };

    let mut approach_x = 0.0f64;
    let mut approach_y = 0.0f64;
    for index in n.saturating_sub(10)..n {
        approach_x += f64::from(prefix[index][0]);
        approach_y += f64::from(prefix[index][1]);
    }
    let approach_norm = norm2(approach_x, approach_y);
    let (approach_cos, approach_sin) = if approach_norm > 1e-6 {
        (
            (approach_x * toward_x + approach_y * toward_y) / approach_norm,
            (approach_x * tangent_x + approach_y * tangent_y) / approach_norm,
        )
    } else {
        (0.0, 0.0)
    };

    for bin in 0..4 {
        let bin_mean = if n == 0 {
            0.0
        } else {
            let low = (n as f64 * bin as f64 / 4.0).floor() as usize;
            let mut high = (n as f64 * (bin + 1) as f64 / 4.0).floor() as usize;
            if high <= low {
                high = (low + 1).min(n);
            }
            if low < n {
                let mut total = 0.0f64;
                for index in low..high {
                    total += speeds[index];
                }
                total / (high - low) as f64
            } else {
                speeds[n - 1]
            }
        };
        out[57 + bin] = bin_mean / mean_speed.max(1e-6);
    }

    let mut along = 0.0f64;
    let mut lateral = 0.0f64;
    for index in shape_start..n {
        let x = f64::from(prefix[index][0]);
        let y = f64::from(prefix[index][1]);
        along += (x * toward_x + y * toward_y).abs();
        lateral += (x * tangent_x + y * tangent_y).abs();
    }
    let lateral_drift = if n > 0 {
        lateral / (along + lateral).max(1e-6)
    } else {
        0.0
    };

    out[51] = curvature_signed;
    out[52] = curvature_absolute;
    out[53] = direction_change_mean;
    out[54] = direction_change_slope;
    out[55] = approach_cos;
    out[56] = approach_sin;
    out[61] = lateral_drift;
    out
}

pub struct SummaryNormalizer {
    pub reorder: Vec<usize>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl SummaryNormalizer {
    pub fn new(reorder: Vec<usize>, mean: Vec<f32>, std: Vec<f32>) -> Self {
        let std = std
            .into_iter()
            .map(|value| if value < 1e-6 { 1.0 } else { value })
            .collect();
        Self { reorder, mean, std }
    }

    pub fn apply(&self, raw: &[f64; SUMMARY_LEN], out: &mut [f32]) {
        for slot in 0..self.reorder.len() {
            let mut value = raw[self.reorder[slot]] as f32;
            if !value.is_finite() {
                value = 0.0;
            }
            let normalized = (value - self.mean[slot]) / self.std[slot];
            out[slot] = if normalized.is_finite() {
                normalized
            } else {
                0.0
            };
        }
    }
}
