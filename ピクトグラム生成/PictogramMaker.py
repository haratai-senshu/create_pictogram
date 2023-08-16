import subprocess
import os
import sys

pathall = os.path.dirname(os.path.abspath(__file__))

while True :
    path_video_1 = input("ファイルのフルパスを入力せいや : ")

    if path_video_1 == "end" :
        sys.exit()
    elif path_video_1 == "remove-completed-video" :
        try :
            os.remove(str(pathall) + '/EX_video/pictogram.mp4')
            print("正常に削除できました。")
        except :
            print("ファイルを削除できません。ファイルが存在しないと思われます。")
    else :
        command = ["python",str(pathall) + "/" + "run6.py",str(path_video_1)]
        proc = subprocess.Popen(command)
        proc.communicate()
        print("")
