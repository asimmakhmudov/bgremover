import sys
import os

from rembg import remove


def write_output(output_data, output_path):
    if hasattr(output_data, 'save'):
        output_data.save(output_path)
        return

    if isinstance(output_data, (bytes, bytearray)):
        with open(output_path, 'wb') as file:
            file.write(output_data)
        return

    raise TypeError('Unsupported output type returned by rembg.remove()')


def remove_background(input_path, output_path):
    try:
        with open(input_path, 'rb') as input_file:
            input_data = input_file.read()

        output_data = remove(input_data)
        if output_data is None:
            raise ValueError('rembg.remove() returned None')

        write_output(output_data, output_path)
        print('SUCCESS')
        return True
    except Exception as exc:
        print(f'ERROR: {exc}')
        return False


def main():
    if len(sys.argv) != 3:
        print('Usage: python model.py <input_path> <output_path>')
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f'ERROR: Input file "{input_path}" does not exist')
        sys.exit(1)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    success = remove_background(input_path, output_path)
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
