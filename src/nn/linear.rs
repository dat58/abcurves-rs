const ROW_TILE: usize = 4;
const COLUMN_TILE: usize = 32;

#[inline]
pub fn matvec(weight: &[f32], rows: usize, columns: usize, input: &[f32], out: &mut [f32]) {
    let input = &input[..columns];
    for row in 0..rows {
        let lane = &weight[row * columns..row * columns + columns];
        let mut total = 0.0f32;
        for column in 0..columns {
            total += lane[column] * input[column];
        }
        out[row] = total;
    }
}

/// Accumulate `value * weights[row]` into `out`, keeping a column tile in
/// registers so the destination is stored once per tile instead of per step.
pub fn accumulate_outer(scales: &[f32], weights: &[f32], stride: usize, out: &mut [f32]) {
    let steps = scales.len();
    let mut column = 0usize;
    while column + COLUMN_TILE <= stride {
        let mut lanes = [0.0f32; COLUMN_TILE];
        lanes.copy_from_slice(&out[column..column + COLUMN_TILE]);
        for step in 0..steps {
            let scale = scales[step];
            let source = &weights[step * stride + column..step * stride + column + COLUMN_TILE];
            for slot in 0..COLUMN_TILE {
                lanes[slot] += scale * source[slot];
            }
        }
        out[column..column + COLUMN_TILE].copy_from_slice(&lanes);
        column += COLUMN_TILE;
    }
    while column < stride {
        let mut total = out[column];
        for step in 0..steps {
            total += scales[step] * weights[step * stride + column];
        }
        out[column] = total;
        column += 1;
    }
}

/// Row and column blocked so each output tile accumulates in registers. The
/// summation order over `inner` is unchanged.
pub fn matmul(
    left: &[f32],
    rows: usize,
    inner: usize,
    right: &[f32],
    columns: usize,
    out: &mut [f32],
) {
    let mut row = 0usize;
    while row < rows {
        let block = ROW_TILE.min(rows - row);
        let mut column = 0usize;
        while column + COLUMN_TILE <= columns {
            let mut lanes = [[0.0f32; COLUMN_TILE]; ROW_TILE];
            for step in 0..inner {
                let source =
                    &right[step * columns + column..step * columns + column + COLUMN_TILE];
                for inside in 0..block {
                    let scale = left[(row + inside) * inner + step];
                    let target = &mut lanes[inside];
                    for slot in 0..COLUMN_TILE {
                        target[slot] += scale * source[slot];
                    }
                }
            }
            for inside in 0..block {
                let at = (row + inside) * columns + column;
                out[at..at + COLUMN_TILE].copy_from_slice(&lanes[inside]);
            }
            column += COLUMN_TILE;
        }
        while column < columns {
            for inside in 0..block {
                let mut total = 0.0f32;
                for step in 0..inner {
                    total += left[(row + inside) * inner + step] * right[step * columns + column];
                }
                out[(row + inside) * columns + column] = total;
            }
            column += 1;
        }
        row += ROW_TILE;
    }
}

#[inline]
pub fn add_bias_rows(values: &mut [f32], bias: &[f32], columns: usize) {
    for row in values.chunks_exact_mut(columns) {
        for column in 0..columns {
            row[column] += bias[column];
        }
    }
}
