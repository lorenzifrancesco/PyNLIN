from enum import Enum

class PulseShape(Enum):
    GAU = 0
    NYQ = 1
    def __str__(self):
        return self.name.lower()   # "gaussian" or "nyquist"

    @classmethod
    def from_str(cls, s: str):
        """Parse from a lowercase string."""
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Unknown pulse shape: {s}")
        
    # specify the line style:
    def line_style(self):
        """Matplotlib line style associated with the pulse shape."""
        if self == PulseShape.GAU:
            return "-"
        elif self == PulseShape.NYQ:
            return "--"
        else:
            raise ValueError(f"Unknown pulse shape: {self}")
