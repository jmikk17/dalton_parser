"""Helper utility functions."""


def get_label(label: str) -> tuple[int, int, str]:
    """Get the atom info from the property label.

    Args:
        label (str): Property label

    Returns:
        list: Atom info in the format [index, nuc_charge, component]

    """
    index = int(label[2:4])
    nuc_charge = int(label[4:6])
    component = label[6:]
    return index, nuc_charge, component
