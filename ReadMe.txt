Данный репозиторий содержит два решения (CPU и GPU):

Для GPU
______________________
(Прежде чем запустить GPU версию нужно убедиться, что на компьютере есть "cuda 9"(И видеокарта от Nvidia не хуже gtx 1060 3gb dual (Не уверен будет ли работать на старых 
карточках, у меня gtx 1060 3gb dual и работает "нормально". Более старые версии видеокарт могут не поддерживаться YOLOv3 плюс, чем лучше видеокарта тем быстрее будет 
обрабатываться видео поток) и установить python версии 3.5.5 и Anaconda c сайта "https://www.anaconda.com/". Затем в консоли анаконды командой 
"conda env create -f environment.yml" установить все необходимые библиотеки для работы алгоритма (Это может занять время))

Для CPU
________________________
Следуйте командам из req.txt

Общее
________________________
Затем скачать по ссылке архив с весами для YOLOv3 и распакованный архив положить в папку с car_counter.py: 
https://yadi.sk/d/Iywdjezjp5nttQ

1) car_counter.py

	Считает машинный трафик. (Кейс)


Чтобы запустить код нужно:
1) в консоли анаконды перейти в папку с people_counter.py (Командой "cd путь")
2) (Для GPU) активировать окружение env3-gpu с установленными библиотеками (Командой "Conda activate env3-gpu")
2) (Для CPU) активировать окружение env-cpu с установленными библиотеками (Командой "Conda activate env-cpu")
3) запустить алгоритм командой ниже:

Для обработки видео
python car_counter.py --input videos/Novgorod_2019-04-05-15_15_00.mp4 --output output/result.avi

Для обработки видео web-камеры в режиме online 
python car_counter.py --output output/result.avi

Важно! Чтобы прервать работу программы нажмите "q"

Список аргументов:
1) Путь к видео
ap.add_argument("-i", "--input", type=str,
	help="path to optional input video file")
2) Путь сохранения результата
ap.add_argument("-o", "--output", type=str,
	help="path to optional output video file")
3) Точность YOLOv3
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
4) Период включения детектора (В фреймах) 
ap.add_argument("-s", "--skip-frames", type=int, default=5,
	help="# of skip frames between detections")
5) Включение горизонтального режима (1) вертикальный режим (0)
ap.add_argument("-a", "--alternative", type=int, default=0,
	help="# of skip frames between detections")
6) Растояние между линиями (Пиксели)
ap.add_argument("-l", "--stepLine", type=int, default=100,
	help="# difference between to lines")
7) Сдвиг линий вверх/вниз
ap.add_argument("-sh", "--shift", type=int, default=0,
	help="# shift of lines")
