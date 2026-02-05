"""
visualize.py

Generates visualizations for the semantic property matching results,
including heatmaps, distribution plots, and summary statistics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_similarity_heatmap(csv_path, save_path):
    """
    Create and save a heatmap of user–property match scores.

    Parameters:
        csv_path (str): Path to match score CSV
        save_path (str): Path to save heatmap image
    """

    # Load results
    df = pd.read_csv(csv_path)

    # Pivot table: users vs properties
    heatmap_data = df.pivot_table(
        index="user_id",
        columns="property_id",
        values="match_score",
        fill_value=0
    )

    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create heatmap using seaborn for better visualization
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Match Score'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title("User–Property Similarity Heatmap", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Property ID", fontsize=12, fontweight='bold')
    plt.ylabel("User ID", fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {save_path}")
    plt.close()


def plot_score_distribution(csv_path, output_dir="outputs/figures"):
    """
    Create and save a histogram of match score distribution.

    Parameters:
        csv_path (str): Path to match score CSV
        output_dir (str): Directory to save figure
    """
    
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(df['match_score'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    
    plt.title("Distribution of Match Scores", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Match Score", fontsize=12, fontweight='bold')
    plt.ylabel("Frequency", fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics
    mean_score = df['match_score'].mean()
    median_score = df['match_score'].median()
    plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    plt.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.2f}')
    plt.legend()
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'score_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Score distribution saved to: {save_path}")
    plt.close()


def plot_user_average_scores(csv_path, output_dir="outputs/figures"):
    """
    Create and save a bar plot of average match score per user.

    Parameters:
        csv_path (str): Path to match score CSV
        output_dir (str): Directory to save figure
    """
    
    df = pd.read_csv(csv_path)
    user_avg = df.groupby('user_id')['match_score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(user_avg.index.astype(str), user_avg.values, edgecolor='black', alpha=0.7)
    
    # Color code by score
    for bar, score in zip(bars, user_avg.values):
        if score >= 80:
            bar.set_color('green')
        elif score >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.title("Average Match Score per User", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("User ID", fontsize=12, fontweight='bold')
    plt.ylabel("Average Match Score", fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'user_average_scores.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ User average scores saved to: {save_path}")
    plt.close()


def plot_property_average_scores(csv_path, output_dir="outputs/figures"):
    """
    Create and save a bar plot of average match score per property.

    Parameters:
        csv_path (str): Path to match score CSV
        output_dir (str): Directory to save figure
    """
    
    df = pd.read_csv(csv_path)
    property_avg = df.groupby('property_id')['match_score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(property_avg.index.astype(str), property_avg.values, edgecolor='black', alpha=0.7)
    
    # Color code by score
    for bar, score in zip(bars, property_avg.values):
        if score >= 80:
            bar.set_color('darkgreen')
        elif score >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('darkred')
    
    plt.title("Average Match Score per Property", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Property ID", fontsize=12, fontweight='bold')
    plt.ylabel("Average Match Score", fontsize=12, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'property_average_scores.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Property average scores saved to: {save_path}")
    plt.close()


def visualize_all(csv_path="outputs/top_k_recommendations.csv", output_dir="outputs/figures"):
    """
    Generate all visualizations from the results CSV.

    Parameters:
        csv_path (str): Path to recommendations CSV
        output_dir (str): Directory to save all figures
    """
    
    print("📊 Generating all visualizations...")
    plot_similarity_heatmap(csv_path, os.path.join(output_dir, 'user_property_heatmap.png'))
    plot_score_distribution(csv_path, output_dir)
    plot_user_average_scores(csv_path, output_dir)
    plot_property_average_scores(csv_path, output_dir)
    print(f"✨ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    visualize_all()
