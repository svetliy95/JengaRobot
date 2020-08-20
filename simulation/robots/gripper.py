import requests
import time
gripper_right_ip = '192.168.10.2'


# see Manual "EGI mit PROFINET-Schnittstelle_de_01.00.pdf" for communication protocol explanation
class OutputData:
    # bits meaning
    # 0  fast stop
    # 1  stop
    # 2  acknowledge
    # 3  prepare for shutdown
    # 4  soft reset
    # 5  release brake
    # 6  reserved
    # 7  grip direction
    # 8  jog mode minus
    # 9  jog mode plus
    # 10 reference
    # 11 release work piece
    # 12 grip work piece
    # 13 move to absolute position
    # 14 move to relative position
    # 15 reserved
    def __init__(self, byte0: str, byte1: str, position, speed, gripping_force):
        assert len(byte0) == 8 and len(byte1) == 8, "Wrong bit string!"
        self.byte0 = byte0
        self.byte1 = byte1
        self.position = position
        self.speed = speed
        self.gripping_force = gripping_force

    def hex(self):
        first_byte = hex(int(self.byte0, 2))[2:].zfill(2)
        second_byte = hex(int(self.byte1, 2))[2:].zfill(2)
        first_doubleword = first_byte + second_byte
        second_doubleword = '0'*4

        # convert position to micrometer and create hex string
        position = int(self.position * 1000)
        position = hex(position)[2:].zfill(8)

        # convert speed to micrometer/s and create hex
        speed = int(self.speed * 1000)
        speed = hex(speed)[2:].zfill(8)

        # convert speed to micrometer/s and create hex
        gripping_force = int(self.gripping_force * 1000)
        gripping_force = hex(gripping_force)[2:].zfill(8)

        # assemble full string
        res = first_doubleword + second_doubleword + position + speed + gripping_force

        return res


# see Manual "EGI mit PROFINET-Schnittstelle_de_01.00.pdf" for communication protocol explanation
class InputData:
    def __init__(self, ip):
        self.byte0 = None
        self.byte1 = None
        self.curent_pos = None
        self.ip = ip

    def get(self):
        url = f'http://{self.ip}/adi/data.json'
        params = {'offset': 0, 'count': 1}
        resp = requests.get(url, params)
        ans = str(resp.json()[0])

        self.byte0 = bin(int(ans[:2], 16))[2:].zfill(8)
        self.byte1 = bin(int(ans[2:4], 16))[2:].zfill(8)
        self.curent_pos = int(ans[4:12], 16) / 1000

    def current_pos(self):
        return self.curent_pos

    def success(self):
        byte0 = self.byte0[::-1]
        return byte0[4] == '1'

    def ready(self):
        byte0 = self.byte0[::-1]
        return byte0[0] == '1'

    def not_feasible(self):
        byte0 = self.byte0[::-1]
        return byte0[3] == '1'

    def warning(self):
        byte0 = self.byte0[::-1]
        return byte0[6] == '1'

    def error(self):
        byte0 = self.byte0[::-1]
        return byte0[7] == '1'

    def no_part_detected(self):
        byte1 = self.byte1[::-1]
        return byte1[3] == '1'

    def gripped(self):
        byte1 = self.byte1[::-1]
        return byte1[4] == '1'

    def position_reached(self):
        byte1 = self.byte1[::-1]
        return byte1[5] == '1'


class Gripper:
    def __init__(self, ip):
        self.ip = ip

    def send_command(self, byte0, byte1, pos, speed, force):
        url = f'http://{self.ip}/adi/update.json'

        value = OutputData(byte0, '00000000', pos, speed, force).hex()
        params = {'inst': 72, 'value': value}

        requests.get(url, params)

        value = OutputData(byte0, byte1, pos, speed, force).hex()
        params = {'inst': 72, 'value': value}

        requests.get(url, params)

    def move_to_pos(self, pos, speed, force=25):
        self.send_command('00000001', '00100000', pos, speed, force)

    def wait_until_positioning_done(self):
        input_data = InputData(self.ip)
        input_data.get()
        while not input_data.success() and not input_data.error() and not input_data.warning() and not input_data.position_reached():
            time.sleep(0.01)
            input_data.get()

        if input_data.success() and input_data.position_reached():
            return True

        if input_data.error() or input_data.warning():
            return False

    def wait_until_gripping_done(self):
        input_data = InputData(self.ip)
        input_data.get()
        while not input_data.success() and \
                not input_data.error() and \
                not input_data.warning() and \
                not input_data.gripped() and \
                not input_data.no_part_detected() and \
                not input_data.not_feasible():
            time.sleep(0.01)
            input_data.get()

        if input_data.success() and input_data.gripped():
            return True

        if input_data.error() or \
                input_data.warning() or \
                input_data.no_part_detected() or \
                input_data.not_feasible():
            return False

    def wait_until_release(self):
        input_data = InputData(self.ip)
        input_data.get()
        while not input_data.success() and \
                not input_data.error() and \
                not input_data.warning() and \
                not input_data.not_feasible():
            time.sleep(0.01)
            input_data.get()

        if input_data.success():
            return True

        if input_data.error() or \
                input_data.warning() or \
                input_data.not_feasible():
            return False

    def grip(self, force=25):
        self.send_command('00000001', '00010000', 0, 200, force)

    def acknowledge_error(self):
        self.send_command('00000101', '00000000', 0, 0, 0)

    def release(self, force=25):
        self.send_command('00000001', '00001000', 0, 200, force)



if __name__ == "__main__":
    g = Gripper('192.168.10.2')
    id = InputData('192.168.10.2')
    velocity = 200

    od = OutputData('00000001', '00000000', 12.4, 200, 25)

    g.move_to_pos(100, velocity)
    g.wait_until_positioning_done()
    id.get()
    print(f"Position reached: {id.position_reached()}")
    print(f"Success: {id.success()}")

    g.grip(25)
    g.wait_until_gripping_done()
    id.get()
    print(f"Success: {id.success()}")
    print(f"Gripped: {id.gripped()}")
    print(f"Not feasible: {id.not_feasible()}")
    print(f"No part detected: {id.no_part_detected()}")
    print(f"Error: {id.error()}")
    print(f"Warning: {id.warning()}")

    g.release(25)
    g.wait_until_release()

