from matplotlib import pyplot as plt


def plot_features(
        df,
        feat_names,
        x_col="start_mtz",  # set to None to use df.index
        figsize_per_row=2.0,
        lw=0.5,
        color="tab:blue",
):
    x = df[x_col] if x_col is not None else df.index
    n = len(feat_names)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        sharex=True,
        figsize=(14, max(2, figsize_per_row * n)),
        constrained_layout=True,
    )

    # Make sure axes is iterable when n == 1
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, feat_names):
        y = df[col].to_numpy(dtype=float)  # NaN -> gaps in line
        ax.plot(x, y, lw=lw, color=color)
        ax.set_ylabel(col)
        ax.set_title(col, loc="left", fontsize=10)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel(x_col if x_col is not None else "index")
    return fig