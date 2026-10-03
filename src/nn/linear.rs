#[inline]
pub fn matvec(weight: &[f32], rows: usize, columns: usize, input: &[f32], out: &mut [f32]) {
    for row in 0..rows {
        let lane = &weight[row * columns..row * columns + columns];
        let mut total = 0.0f32;
        for column in 0..columns {
            total += lane[column] * input[column];
        }
        out[row] = total;
    }
}

#[inline]
pub fn matmul(
    left: &[f32],
    rows: usize,
    inner: usize,
    right: &[f32],
    columns: usize,
    out: &mut [f32],
) {
    for row in 0..rows {
        let source = &left[row * inner..row * inner + inner];
        let target = &mut out[row * columns..row * columns + columns];
        target.fill(0.0);
        for step in 0..inner {
            let scale = source[step];
            let lane = &right[step * columns..step * columns + columns];
            for column in 0..columns {
                target[column] += scale * lane[column];
            }
        }
    }
}
