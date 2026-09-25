# Одна прочитанная этикетка на кадре
std_msgs/Header header            # время кадра, frame_id камеры
string text                       # декодированное содержимое
string symbology                  # Code 128, DataMatrix, QRCode...
geometry_msgs/Point[4] corners    # углы в пикселях (x=u, y=v, z=0)
string sscc                       # SSCC, если распознан GS1 AI (00) с верной контрольной цифрой
string address                    # адрес ячейки после привязки; пусто — не привязана
float32 confidence
