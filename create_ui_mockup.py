#!/usr/bin/env python3
"""
Create a visual representation of the AI Assistant interface
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_ui_mockup():
    """Create a mockup of the AI Assistant interface"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, '🤖 AI Privacy Assistant', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Risk Assessment Box
    risk_box = FancyBboxPatch((0.5, 9), 9, 1.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#ffebee', 
                              edgecolor='#f44336', 
                              linewidth=2)
    ax.add_patch(risk_box)
    ax.text(5, 10.2, '🚨 Privacy Risk Assessment', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(1, 9.8, '• 🔴 High likelihood of extensive data collection', 
            ha='left', va='center', fontsize=10)
    ax.text(1, 9.5, '• 🔴 High likelihood of data sharing with third parties', 
            ha='left', va='center', fontsize=10)
    ax.text(1, 9.2, '• 🟡 Some unclear language detected', 
            ha='left', va='center', fontsize=10)
    
    # Suggested Questions Box
    questions_box = FancyBboxPatch((0.5, 6.5), 9, 2, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#e3f2fd', 
                                   edgecolor='#2196f3', 
                                   linewidth=2)
    ax.add_patch(questions_box)
    ax.text(5, 8.2, '💭 Suggested Questions', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Question buttons (2 columns)
    questions = [
        "❓ What data does this policy collect?",
        "❓ Can I delete my personal information?",
        "❓ Does this app share data with third parties?", 
        "❓ How long do they keep my data?"
    ]
    
    for i, question in enumerate(questions):
        x = 1.5 if i % 2 == 0 else 5.5
        y = 7.7 - (i // 2) * 0.4
        
        button = FancyBboxPatch((x-0.3, y-0.15), 3.6, 0.3, 
                                boxstyle="round,pad=0.05", 
                                facecolor='#bbdefb', 
                                edgecolor='#1976d2')
        ax.add_patch(button)
        ax.text(x + 1.5, y, question[:35] + "...", 
                ha='center', va='center', fontsize=8)
    
    # Chat Interface Box
    chat_box = FancyBboxPatch((0.5, 3.5), 9, 2.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#f3e5f5', 
                              edgecolor='#9c27b0', 
                              linewidth=2)
    ax.add_patch(chat_box)
    ax.text(5, 5.7, '💬 Ask the AI Assistant', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Text input field
    input_box = FancyBboxPatch((1, 5), 8, 0.4, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor='#666')
    ax.add_patch(input_box)
    ax.text(1.2, 5.2, 'e.g., What personal data does this app collect?', 
            ha='left', va='center', fontsize=9, style='italic', color='#666')
    
    # Action buttons
    ask_btn = FancyBboxPatch((1, 4.3), 2, 0.4, 
                             boxstyle="round,pad=0.05", 
                             facecolor='#4caf50', 
                             edgecolor='#388e3c')
    ax.add_patch(ask_btn)
    ax.text(2, 4.5, '🚀 Ask Assistant', 
            ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    
    clear_btn = FancyBboxPatch((3.5, 4.3), 1.5, 0.4, 
                               boxstyle="round,pad=0.05", 
                               facecolor='#ff9800', 
                               edgecolor='#f57c00')
    ax.add_patch(clear_btn)
    ax.text(4.25, 4.5, '🧹 Clear', 
            ha='center', va='center', fontsize=9, color='white')
    
    history_btn = FancyBboxPatch((5.5, 4.3), 1.5, 0.4, 
                                 boxstyle="round,pad=0.05", 
                                 facecolor='#607d8b', 
                                 edgecolor='#455a64')
    ax.add_patch(history_btn)
    ax.text(6.25, 4.5, '🔄 History', 
            ha='center', va='center', fontsize=9, color='white')
    
    # Response Box
    response_box = FancyBboxPatch((0.5, 0.5), 9, 2.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='#e8f5e8', 
                                  edgecolor='#4caf50', 
                                  linewidth=2)
    ax.add_patch(response_box)
    ax.text(5, 2.7, '🤖 Assistant Response', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax.text(1, 2.3, '❓ Question: What personal data does this app collect?', 
            ha='left', va='center', fontsize=10, fontweight='bold')
    
    response_text = ("💡 Answer: Based on the privacy policy analysis, this app collects several "
                     "types of personal data including your name, email address, location data, "
                     "and browsing behavior. The policy indicates this data is collected to provide "
                     "personalized services, but you should be aware that it may also be shared "
                     "with third-party partners for marketing purposes.")
    
    # Wrap text
    words = response_text.split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        if len(' '.join(current_line)) > 80:
            lines.append(' '.join(current_line[:-1]))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    for i, line in enumerate(lines[:4]):  # Show only first 4 lines
        ax.text(1, 2 - i*0.2, line, 
                ha='left', va='center', fontsize=9)
    
    plt.title('AI Privacy Assistant Interface Preview', 
              pad=20, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/tmp/ai_assistant_mockup.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print("✅ UI mockup saved to /tmp/ai_assistant_mockup.png")

if __name__ == "__main__":
    create_ui_mockup()