import torch

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as Colormap
from matplotlib.colors import LogNorm

def visualize_attention(multihead_attention, output_path="atten_map_1.png", v_mask=None):
    """
        from FastV code github [https://github.com/pkunlp-icler/FastV/blob/main/src/FastV/inference/plot_inefficient_attention.py#L246]
        multihead_attention: (num_heads, n_tokens, n_tokens)
        v_mask: (n_tokens), bool, 1 for visual tokens 0 for text tokens
    """
    
    averaged_attention = torch.mean(multihead_attention, axis=0).float() # Shape: (n_tokens, n_tokens)
    
    # pooling the attention scores  with stride 20
    if v_mask is not None:
        averaged_attention = averaged_attention[~v_mask][:, v_mask]
        #! Prefill do not require this
        # rows = (~v_mask).nonzero().flatten().unsqueeze(1)
        # cols = v_mask.nonzero().flatten().unsqueeze(0)
        # mask_upper = rows < cols
        # Mask out the upper half (above the main diagonal)
        # For instance, set them to 0.0
        # averaged_attention[mask_upper] = 0.0
    # averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), , stride=20).squeeze(0).squeeze(0)
    
    cmap = plt.cm.get_cmap("viridis")
    plt.figure(figsize=(5, 5),dpi=400)
    
    # set the x and y ticks to 20x of the original
    ax = sns.heatmap(
        averaged_attention,
        cmap=cmap,
        # norm=LogNorm(),
        # cbar_kws={'label': 'Attention Score (Log scale)'},
        )
    
    # remove the x and y ticks & replace the x and y ticks with string
    # x_skip_factor = 4
    # y_skip_factor = 1
    # x_ticks = [str(i * x_skip_factor) for i in range(0, averaged_attention.shape[0], x_skip_factor)]
    # y_ticks = [str(i * y_skip_factor) for i in range(0, averaged_attention.shape[1], y_skip_factor)]
    # ax.set_xticks([i for i in range(0, averaged_attention.shape[0], x_skip_factor)])
    # ax.set_yticks([i for i in range(0, averaged_attention.shape[1], y_skip_factor)])
    # ax.set_xticklabels(x_ticks)
    # ax.set_yticklabels(y_ticks)

    # change the x & y ticks font size
    # plt.xticks(fontsize = 3)
    # plt.yticks(fontsize = 3)
    plt.xlabel("Visual Tokens (Key)")
    plt.ylabel("Text Tokens (Query)")
    # make y label vertical
    # plt.yticks(rotation = 0)
    # plt.xticks(rotation = 90)     
    
    # plt.title(title)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    top_five_attentions = []
    for row in averaged_attention:
        # Use torch.topk to get the top 5 values and their indices
        top_values, top_indices = torch.topk(row, 10)
        # Convert to lists and append to the overall list
        top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
        top_five_attentions.append(top_five_line)
        
    return top_five_attentions,averaged_attention    