import socket
import time

# Echo client program
import socket

HOST = '192.168.10.103'    # The remote host
PORT = 10002              # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print('Connected!')
# s.sendall('Hello, world')
msg = '(343.16,-276.15,576.61,-179.89,-0.01,-46.79)'
for i in range(1):
    s.send(msg.encode('ascii'))
    data = s.recv(1024)
    print(f'Received: {str(data.decode("ascii"))}')
    time.sleep(0.1)
s.close()