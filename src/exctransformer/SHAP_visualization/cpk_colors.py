import colorsys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


class HsbColor:
    def __init__(self, hue, saturation, brightness):
        """
        Initialize HSB color.
        
        Parameters:
          hue         : Hue (0 to 255)
          saturation  : Saturation (0 to 255)
          brightness  : Brightness (0 to 255)
        """
        self.hue = hue
        self.saturation = saturation
        self.brightness = brightness

    def to_rgb(self):
        """
        Converts the HSB color to an RGB tuple.
        The conversion normalizes the HSB values to [0, 1] for colorsys.
        Returns:
          (R, G, B) where each component is an integer in [0, 255].
        """
        h = self.hue / 255.0
        s = self.saturation / 255.0
        v = self.brightness / 255.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def __repr__(self):
        return f"HsbColor(hue={self.hue}, sat={self.saturation}, bri={self.brightness}, RGB={self.to_rgb()})"

class Item:
    # Define an expected maximum number of items for non-carbon group hue scaling.
    EXPECTED_MAX = 15
    # Calculate the hue factor.
    HUE_FACTOR = 255 / EXPECTED_MAX

    def __init__(self, id):
        self.id = id

    @property
    def color(self):
        """
        Compute the HSB color for this item based on its ID.
        
        Special handling for the carbon group (atomic numbers 6, 14, 32, 50, 82):
          - Carbon (6) is always black.
          - Other elements in the group are rendered as dark gray shades,
            with brightness increasing slightly with group order.
        
        For all other elements, a golden-ratio–based algorithm is used:
          - Computes a base hue from the ID.
          - Adjusts brightness and saturation if the hue is red-ish or blue-ish.
        """
        # Define the carbon group.
        carbon_group = [6, 14, 32, 50, 82]
        if self.id in carbon_group:
            if self.id == 6:
                # Carbon is always black.
                return HsbColor(0, 0, 0)
            else:
                # For other group members, assign a dark gray shade.
                # Map group order to brightness: for silicon (14) use 30,
                # germanium (32) use 60, tin (50) use 90, lead (82) use 120.
                idx = carbon_group.index(self.id)
                brightness = 30 + (idx - 1) * 30
                return HsbColor(0, 0, brightness)
        else:
            # For all other elements, use the golden-ratio based approach.
            hue = int((self.id * Item.HUE_FACTOR) % 255)
            # Convert hue to degrees (0-360) for conditional adjustment.
            deg = hue * (360.0 / 255.0)
            # Default saturation and brightness.
            sat = 175
            bri = 175
            # If hue is in a red-ish range (<30° or >330°) or blue-ish (200° to 260°),
            # darken and desaturate to yield a more grayish appearance.
            if deg < 30 or deg > 330 or (200 <= deg <= 260):
                bri = int(bri * 0.5)
                sat = int(sat * 0.7)
            return HsbColor(hue, sat, bri)

    def __repr__(self):
        return f"Item(id={self.id}, color={self.color})"

def get_cpk_color(atomic_number):
    """
    Returns a CPK color for a given atomic number using our HSB-based algorithm.
    
    The function creates an Item for the atomic number and returns its color as an RGB tuple
    with components normalized to [0,1].
    """
    item = Item(atomic_number)
    rgb = item.color.to_rgb()
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

# class CPKColorAssigner:
#     def __init__(self, default_assignments=None):
#         """
#         Initializes the CPK color assigner with a fixed palette.
        
#         Parameters:
#           - default_assignments: Optional list of dictionaries with atomic number keys and
#                                  hex color values. These defaults are applied immediately.
#                                  Example: [{6: '#000000'}, {8: '#FFA500'}]
#         """
#         self.palette = [
#             ('Black', '#000000'),
#             ('Orange', '#FFA500'),
#             ('Green', '#008000'),
#             ('Violet', '#8A2BE2'),
#             ('Gold', '#FFD700'),
#             ('Lavender', '#E6E6FA'),  # Replacement for Silver
#             ('Gray', '#808080'),
#             ('Brown', '#A52A2A'),
#             ('Turquoise', '#40E0D0'),
#             ('Yellow', '#FFFF00'),
#             ('Olive', '#808000'),
#             ('Copper', '#B87333'),
#             ('Coral', '#FF7F50'),
#             ('Peach', '#FFDAB9'),
#             ('Cream', '#FFFDD0')
#         ]
#         # Set of allowed hex colors.
#         self.allowed_hex = {hex_color for _, hex_color in self.palette}
#         # Mapping of atomic numbers to assigned hex colors.
#         self.assigned_colors = {}
        
#         # Apply any provided default assignments.
#         if default_assignments is not None:
#             self.apply_default_assignments(default_assignments)
    
#     def apply_default_assignments(self, default_assignments):
#         """
#         Processes a list of default assignments. Each element in the list should be a dictionary
#         with atomic numbers (or keys convertible to int) as keys and allowed hex colors as values.
#         """
#         if not isinstance(default_assignments, list):
#             raise ValueError("default_assignments must be a list of dictionaries.")
        
#         for default_dict in default_assignments:
#             if not isinstance(default_dict, dict):
#                 raise ValueError("Each default assignment must be a dictionary.")
#             for key, color in default_dict.items():
#                 try:
#                     atomic_num = int(key)
#                 except ValueError:
#                     raise ValueError("Atomic number keys in default_assignments must be integers or convertible to integers.")
#                 if color not in self.allowed_hex:
#                     raise ValueError(f"Color {color} is not within the allowed palette.")
#                 self.assigned_colors[atomic_num] = color

#     def get_color(self, atomic_number, set_color=None, default_assignments=None):
#         """
#         Returns a CPK color (hex string) for the given atomic number.
        
#         Behavior:
#           - If default_assignments is provided (a list of dicts), they are applied first.
#           - If set_color is provided, it forces the assignment (if allowed).
#           - If the atomic number is already assigned, that color is returned.
#           - Otherwise, the next available color from the palette is assigned.
#           - Raises an error if no colors remain.
        
#         Parameters:
#           - atomic_number : int
#           - set_color     : Optional hex color to force the assignment.
#           - default_assignments: Optional list of dictionaries (same format as for __init__)
        
#         Example usage:
#           >>> assigner.get_color(6, default_assignments=[{6: '#000000'}])
#           >>> assigner.get_color(8, set_color='#FFA500', default_assignments=[{6: '#000000'}, {8: '#FFA500'}])
#         """
#         # If new default assignments are provided, apply them.
#         if default_assignments is not None:
#             self.apply_default_assignments(default_assignments)
        
#         # If a forced color is provided, assign and return it.
#         if set_color is not None:
#             if set_color not in self.allowed_hex:
#                 raise ValueError(f"Color {set_color} is not within the allowed palette.")
#             self.assigned_colors[atomic_number] = set_color
#             return set_color
        
#         # Return the already assigned color if available.
#         if atomic_number in self.assigned_colors:
#             return self.assigned_colors[atomic_number]
        
#         # Compute the next available color by iterating through the palette.
#         for _, hex_color in self.palette:
#             if hex_color not in self.assigned_colors.values():
#                 self.assigned_colors[atomic_number] = hex_color
#                 return hex_color
        
#         raise RuntimeError("Exceeded the number of available colors in the palette.")




# def get_cpk_color(atomic_number):
#     """
#     Returns a unique CPK color for a given atomic number using a 
#     golden ratio based algorithm to spread the colors over the allowed hue span.
    
#     Allowed hues:
#       - 15° to 225° (span = 210°)
#       - 255° to 345° (span = 90°)
#     Total allowed span: 300°.
    
#     The function computes a fraction using the golden ratio conjugate,
#     scales that fraction to [0, 300] and then maps it into the two allowed segments.
#     Saturation and value are fixed at 0.8 for vivid colors.
#     """
#     # Golden ratio conjugate (≈0.618) helps in generating well-spaced values.
#     golden_ratio_conjugate = 0.618033988749895
    
#     # Use the atomic number to generate a pseudo-random fraction.
#     fraction = (atomic_number * golden_ratio_conjugate) % 1  # value in [0,1)
    
#     # Scale fraction to the total allowed hue span (300°).
#     allowed_hue = fraction * 300  # value in [0,300)
    
#     # Map allowed_hue into the two allowed segments:
#     # If allowed_hue is less than 210, it goes into the first segment (15° to 225°).
#     # Otherwise, it maps into the second segment (255° to 345°).
#     if allowed_hue < 210:
#         hue = 15 + allowed_hue
#     else:
#         hue = 255 + (allowed_hue - 210)
    
#     # Convert hue (in degrees) to RGB with fixed saturation and value.
#     saturation = 0.8
#     value = 0.8
#     r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    
#     return (round(r, 2), round(g, 2), round(b, 2))

# cpk_colors = {
#     1: (1.00, 1.00, 1.00),  # Hydrogen - White
#     2: (0.85, 1.00, 1.00),  # Helium - Light Cyan
#     3: (0.80, 0.50, 1.00),  # Lithium - Light Purple
#     4: (0.76, 1.00, 0.00),  # Beryllium - Light Green
#     5: (1.00, 0.71, 0.71),  # Boron - Pink
#     6: (0.56, 0.56, 0.56),  # Carbon - Gray
#     7: (0.19, 0.31, 0.97),  # Nitrogen - Blue
#     8: (1.00, 0.05, 0.05),  # Oxygen - Red
#     9: (0.56, 0.88, 0.31),  # Fluorine - Green
#     10: (0.70, 0.89, 0.96), # Neon - Light Blue
#     11: (0.67, 0.36, 0.95), # Sodium - Purple
#     12: (0.54, 1.00, 0.00), # Magnesium - Green
#     13: (0.75, 0.65, 0.65), # Aluminum - Silver
#     14: (0.94, 0.78, 0.63), # Silicon - Orange
#     15: (1.00, 0.50, 0.00), # Phosphorus - Orange
#     16: (1.00, 1.00, 0.19), # Sulfur - Yellow
#     17: (0.12, 0.94, 0.12), # Chlorine - Green
#     18: (0.50, 0.82, 0.89), # Argon - Light Blue
#     19: (0.56, 0.25, 0.83), # Potassium - Purple
#     20: (0.24, 1.00, 0.00), # Calcium - Green
#     26: (0.87, 0.40, 0.20), # Iron - Orange
#     29: (1.00, 0.47, 0.25), # Copper - Orange
#     47: (0.75, 0.75, 0.75), # Silver - Gray
#     79: (1.00, 0.84, 0.00), # Gold - Gold
# }

# # Default color for unknown elements
# default_color = (0.50, 0.50, 0.50)  # Gray

# def get_cpk_color(atomic_number):
#     """Returns the CPK color for a given atomic number, or a default color if not found."""
#     return cpk_colors.get(atomic_number, default_color)
