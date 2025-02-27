import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_graph(data, step=None, wseg=False, walign=False):
    """Generates and saves a line plot for the given hyperparameter data."""

    # Ensure output directory exists
    os.makedirs("figures", exist_ok=True)
    
    df = pd.DataFrame(data)
    
    # Extract data
    hyperparam = df.columns[0]
    label1, label2, label3= df.columns[1:4]
    x = df[hyperparam]
    y1, y2, y3 = df[label1], df[label2], df[label3]

    if step is not None:
        x_ticks = np.arange(min(x), max(x) + step, step)  # Generate x-axis ticks
    else:
        x_ticks = x

    # Plot setup
    plt.figure(figsize=(6, 4))
    axisfont=25
    legendfont=20
    tickfont=20

    if wseg or walign:
        plt.xscale('log')  # Set x-axis to log scale
        plt.minorticks_off()  # Disable extra minor ticks

    plt.plot(x, y1, marker='o', markersize=12, linewidth=5, label=label1, color='#5784F5')
    plt.plot(x, y2, marker='o', markersize=12, linewidth=5, label=label2, color='#DE4036')
    plt.plot(x, y3, marker='o', markersize=12, linewidth=5, label=label3, color='#F3BC00')

    # Configure axes
    if wseg or walign:
        plt.xlabel(hyperparam, fontsize=axisfont)
        plt.xticks(x_ticks, fontsize=tickfont)
    else:
        plt.xlabel(hyperparam, fontsize=axisfont)
        plt.xticks(x_ticks, fontsize=tickfont)
    
    # plt.ylabel('Metrics', fontsize=15)
    
    if walign:
        plt.yticks(np.arange(0,1.1,0.2), fontsize=tickfont)
        # plt.ylim(0.4, 1.0)
    else:
        plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=tickfont)
        plt.ylim(0.5, 1.0)
    
    plt.tick_params(axis='y', length=0, width=0)

    # Customize spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)

    # Grid & Legend
    plt.grid(axis='both', linestyle='-', alpha=0.7)
    plt.legend(fontsize=legendfont, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3
               ,frameon=False, borderpad=-0.25, handlelength=0)

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"figures/{hyperparam}.pdf")
    plt.close()
    print(f"Figure Saved at figures/{hyperparam}")


def hyperparam_graphs():
    for i in range(5):
        # Define data and corresponding steps
        steps = [0.01, 0.1, 0.05, 0.01, 0.125]

        data = [
            {  # GW radius r
                "r": [0.02, 0.03, 0.04, 0.05, 0.06],
                'Acc@1.0': [0.9463, 0.9444, 0.9484, 0.9354, 0.9501],
                'Progress': [0.9163, 0.8813, 0.9309, 0.8705, 0.8855],
                'τ': [0.8828, 0.7848, 0.8415, 0.7910, 0.7612]
            },
            {  # GW weight α
                "α": [0.1, 0.2, 0.3, 0.4, 0.5],
                'Acc@1.0': [0.9488, 0.9412, 0.9463, 0.9392, 0.9380],
                'Progress': [0.9279, 0.8936, 0.9163, 0.9122, 0.9009],
                'τ': [0.8697, 0.7893, 0.8828, 0.8272, 0.8301]
            },
            {  # Global temporal prior ρ
                "ρ": [0.25, 0.30, 0.35, 0.40, 0.45],
                'Acc@1.0': [0.9408, 0.9380, 0.9463, 0.9436, 0.9374],
                'Progress': [0.9157, 0.9212, 0.9163, 0.9245, 0.9047],
                'τ': [0.8385, 0.8220, 0.8828, 0.8526, 0.8554]
            },
            {  # Entropy reg. ε
                "ε": [0.05, 0.06, 0.07, 0.08, 0.09],
                'Acc@1.0': [0.9430, 0.9402, 0.9463, 0.9410, 0.9450],
                'Progress': [0.9177, 0.8903, 0.9163, 0.9362, 0.9222],
                'τ': [0.8130, 0.8400, 0.8828, 0.8777, 0.8585]
            },
            {  # Virtual frame threshold ζ
                "ζ": [0.25, 0.375, 0.5, 0.625, 0.75],
                'Acc@1.0': [0.9414, 0.9382, 0.9463, 0.9434, 0.9442],
                'Progress': [0.9098, 0.9101, 0.9163, 0.9070, 0.8957],
                'τ': [0.8131, 0.8548, 0.8828, 0.8096, 0.7596]
            }
        ]

        plot_graph(data=data[i], step=steps[i])

def joint_model_graphs():
        joint_align_model_data={
            r'$w_{seg}$': [0.01, 0.1, 1, 10, 100],
            'Acc@1.0': [0.9471, 0.9449, 0.9325, 0.9290, 0.8871],
            'Progress': [0.9184, 0.9227, 0.9171, 0.9187, 0.8985],
            'τ': [0.8604, 0.8585, 0.8864, 0.8765, 0.8231]
        }
        joint_seg_model_data= {
            r'$w_{align}$': [0.01, 0.1, 1, 10, 100],
            "F1": [0.6488, 0.6381, 0.7514, 0.7352, 0.7376],
            "MoF": [0.6147, 0.6384, 0.6989, 0.6818, 0.6856],
            "mIoU": [0.4139, 0.4325, 0.4925, 0.5011, 0.5090]
        }
        plot_graph(data=joint_align_model_data, wseg=True)
        plot_graph(data=joint_seg_model_data, walign=True)


if __name__ == "__main__":
    hyperparam_graphs()
    joint_model_graphs()