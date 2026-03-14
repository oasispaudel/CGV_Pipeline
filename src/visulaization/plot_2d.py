import matplotlib.pyplot as plt


def draw_lines(ax, lines, color, title):
    for line in lines:
        x1, y1, x2, y2 = line
        ax.plot([x1, x2], [y1, y2], marker="o", color=color)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)


def visualize_2d_pipeline(original, transformed, clipped, final_output):

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    draw_lines(axs[0, 0], original, "red", "Original Object")
    draw_lines(axs[0, 1], transformed, "blue", "After Transformation")
    draw_lines(axs[1, 0], clipped, "green", "After Clipping")
    draw_lines(axs[1, 1], final_output, "purple", "Final Viewport Output")

    plt.tight_layout()
    plt.show()