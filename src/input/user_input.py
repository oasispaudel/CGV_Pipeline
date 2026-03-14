def choose_dimension():
    print("Choose object dimension:")
    print("1. 2D")
    print("2. 3D")

    choice = input("Enter choice: ")

    if choice == "1":
        return "2D"
    elif choice == "2":
        return "3D"
    else:
        print("Invalid choice, defaulting to 2D")
        return "2D"


def choose_projection():
    print("\nChoose projection type:")
    print("1. Orthographic")
    print("2. Perspective")

    choice = input("Enter choice: ")

    if choice == "1":
        return "orthographic"
    elif choice == "2":
        return "perspective"
    else:
        print("Invalid choice, defaulting to orthographic")
        return "orthographic"


def get_line_input():
    print("\nEnter line coordinates")

    x1 = float(input("x1: "))
    y1 = float(input("y1: "))
    x2 = float(input("x2: "))
    y2 = float(input("y2: "))

    return [(x1, y1, x2, y2)]