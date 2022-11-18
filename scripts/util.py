def printWithPadding(string: str, qty=100) -> None:
    """Prints a string with equals padding on either side."""
    qty = qty - qty % 2
    padding = "=" * ((qty - len(string)) // 2)
    print(f"{padding}{string}{padding}")
