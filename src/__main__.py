from src.main import main

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as error:
        print(f"An unexpected error ocurred {type(error).__name__}")
