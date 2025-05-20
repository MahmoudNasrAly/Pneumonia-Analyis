import matplotlib.pyplot as plt
from matplotlib import font_manager
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np

def setup_arabic_font():
    """
    Set up Arabic font for matplotlib plots
    """
    # Try to use a system Arabic font
    arabic_fonts = [
        'Arial',
        'Times New Roman',
        'Tahoma',
        'Microsoft Sans Serif',
        'Segoe UI'
    ]
    
    for font in arabic_fonts:
        try:
            font_manager.findfont(font)
            plt.rcParams['font.family'] = font
            break
        except:
            continue

def arabic_text(text):
    """
    Convert Arabic text to proper display format
    """
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

def plot_with_arabic_text(ax, x, y, title, xlabel, ylabel):
    """
    Create a plot with proper Arabic text handling
    """
    ax.plot(x, y)
    ax.set_title(arabic_text(title))
    ax.set_xlabel(arabic_text(xlabel))
    ax.set_ylabel(arabic_text(ylabel))
    return ax

# Example usage:
if __name__ == "__main__":
    setup_arabic_font()
    
    # Example plot with Arabic text
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plot_with_arabic_text(
        ax,
        x, y,
        "رسم بياني للدالة الجيبية",
        "المحور السيني",
        "المحور الصادي"
    )
    
    plt.show() 