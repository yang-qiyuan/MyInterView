def rref(matrix):
    # reduced row echelon form algorithm
    # 找主元， 最好找最大值
    # 交换
    # 归一化
    # 消元
    matrix = matrix.astype(np.float16)
    m, n = matrix.shape
    r = 0
    # iterate over the columns
    for c in range(n):
        # find the pivot
        nonzero_id = np.nonzero(matrix[r:, c])[0]
        if len(nonzero_id) == 0:
            continue
        pivot = nonzero_id[0] + r
        # swap with the current row r
        matrix[[pivot, r]] = matrix[[r, pivot]]
        # nomarlize by the pivot
        matrix[r] = matrix[r]/ matrix[r, c]
        # eliminate other elements in that column
        for i in range(m):
            if i != r:
                matrix[i] -= matrix[i, c]*matrix[r]
        r += 1 # next column
        if r == m: break
    return matrix