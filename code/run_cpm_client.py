# -*-coding:Utf-8 -*
# ------------------------------------------------------------------ #
# To use this routine:
# 1/ Run the server.
#    Method one:
#       python run_cpm.py open-port
#    Method two (recommended):
#       python run_cpm_client.py open-port
# 2/ Run the client from any modeling code with, e.g.
#       python run_cpm_client.py 200069974 92 800 1e3 0 16 5 ./tpf ./output/200069974 -p ./test_pixel.dat
#    using the relevant parameters.
# 3/ At the very end of the modeling process, close the port and kill the server.
#    Method one: use Ctrl+C in the terminal where server is running.
#    Method two (recommended):
#       python run_cpm_client.py close-port
# ------------------------------------------------------------------ #
# Packages
# ------------------------------------------------------------------ #
import os
import sys
import socket
# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
if __name__ == '__main__':
    fn_portnumber = 'portnumber.txt'
    # User ask to mimic a server. Opening a port.
    if sys.argv[1].strip() == 'open-port':
        if not os.path.exists(fn_portnumber):
            command = 'python run_cpm_dave_server.py open-port > cpm_routine_outputs.txt &'
            os.system(command)
            # Wait for the server is ready
            while not os.path.exists(fn_portnumber):
                continue
    else:
        # Identify which PORT should use (same as server)
        if os.path.exists(fn_portnumber):
            file = open(fn_portnumber, 'r')
            PORT = int(file.readline().strip())
            file.close()
        else:
            sys.exit('PORT number cannot be found.')

        # Open a connection with server and send arguments to CPM routine
        sck = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sck.connect(('', PORT))
        sck.sendall(' '.join(sys.argv))
        sck.close()
