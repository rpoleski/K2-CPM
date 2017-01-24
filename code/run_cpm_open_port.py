
import socket
import os

import run_cpm


fn_portnumber = "portnumber.txt"


def close_socket_remove_file(socket, file_name=fn_portnumber, message="\nServer closed by user."):
    """Close the port and kill the server."""
    socket.close()
    os.remove(file_name)
    if message is not None:
        print(message)


def open_port():
    sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Find available PORT number > 5999
    port = 5999
    while True:
        port += 1
        try:
            sck.bind(('', port))
        except:
            continue
        break
    # Identify which PORT is opened and save it
    port = sck.getsockname()[1]
    with open(fn_portnumber, "w") as f:
        f.write("{:d}".format(port))
    # Server wait for a client request
    print('Current directory:\n{:s}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('Opening PORT {:d}'.format(port))
    print("The server is ready. To stop it, press Ctrl+C or call the client with the argument 'stop'.")
    sck.listen(1)
    try:
        while True:
            (conn, address) = sck.accept()
            command = conn.recv(1024)  # 1024 is the maximum size of the arguments.
            if not command:
                break
            if command.split(' ')[-1]=='close-port':
                close_socket_remove_file(socket=sck)
                break
            else: # Run CPM.
                sys.argv = command.split(' ') 
                run_cpm.main() 
    except KeyboardInterrupt: # If user hit Ctrl+C.
        msg = "\nServer killed. All the related processes will be killed as well."
        close_socket_remove_file(socket=sck, message=msg)


if __name__ == '__main__':
   open_port()

