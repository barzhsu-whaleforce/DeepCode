"""
DeepCode - AI Research Engine

🧬 Next-Generation AI Research Automation Platform
⚡ Transform research papers into working code automatically
"""

__version__ = "1.0.1"
__author__ = "DeepCode Team"
__url__ = "https://github.com/HKUDS/DeepCode"
__description__ = "AI Research Engine - Transform research papers into working code automatically"

# Import main components for easy access
from .utils import FileProcessor, DialogueLogger

__all__ = [
    "FileProcessor", 
    "DialogueLogger",
    "__version__",
    "__author__",
    "__url__",
    "__description__"
] 