import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def draw_3d_lines(ax, lines, color, title):

    for line in lines:
        x1, y1, z1, x2, y2, z2 = line

        ax.plot(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            marker="o",
            color=color
        )

    ax.set_title(title)


def visualize_3d_pipeline(original, transformed, clipped, final_output):

    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

    draw_3d_lines(ax1, original, "red", "Original Object")
    draw_3d_lines(ax2, transformed, "blue", "After Transformation")
    draw_3d_lines(ax3, clipped, "green", "After Clipping")
    draw_3d_lines(ax4, final_output, "purple", "Final Output")

    plt.tight_layout()
    plt.show()