# Prototipe monitor masker dan hairnet

Prototipe untuk maksimal enam sumber RTSP di jaringan lokal: tampilan langsung dan 100 kejadian terakhir. Semua pengguna memakai satu akun bersama. Tidak ada pengenalan identitas; foto perlu ditinjau petugas. Tanpa model yang sesuai, video tetap bisa tampil tetapi **tidak ada deteksi atau kejadian yang dibuat**. Pemeriksaan cloud Gemini bersifat opsional dan dinonaktifkan secara default.

Tampilan langsung diregangkan untuk memenuhi kotak 16:9 tanpa memotong gambar. Jika rasio sumber berbeda, proporsi objek berubah di tampilan; foto kejadian tetap menggunakan frame sumber utuh dengan proporsi asli.

Halaman menampilkan video Kamera 1 di kiri dan dua tabel riwayat di kanan: tangkapan semua orang di atas, indikasi tanpa masker dari Gemini di bawah. Masing-masing menampilkan **6 tangkapan terbaru**, paling baru di atas berdasarkan waktu gambar, lalu ID rekaman sebagai pembeda jika waktunya sama. Semua tangkapan tetap disimpan; batas enam hanya membatasi tampilan, bukan menghapus data. Gambar diambil lokal dari frame asli tanpa anotasi setelah YOLO menemukan kotak `person`, tanpa menunggu atau memerlukan Gemini. Semua kotak orang pada sampel disertakan. Tangkapan pertama langsung tersedia, lalu disampel setiap 3 detik atau ketika jumlah kotak berubah. Tabel menampilkan ID rekaman, waktu tangkapan, potongan orang, wajah/area kepala, dan skor deteksi orang (bukan keyakinan masker). ID adalah ID tangkapan, bukan identitas orang; orang yang sama dapat tercatat berulang.

Detektor wajah frontal Haar bawaan OpenCV mencari wajah pada bagian atas potongan orang. Wajah yang ditemukan ditampilkan sebagai crop terpisah; jika tidak ditemukan atau ada beberapa kandidat, tabel menampilkan area kepala dengan label yang jelas, bukan mengklaim wajah berhasil terdeteksi. Detektor ini tidak mengenali identitas dan dapat keliru pada wajah kecil, tertutup masker, membelakangi kamera, atau kabur. Thumbnail orang dibatasi sisi panjang 512 piksel dan wajah/kepala 160 piksel, tanpa memperbesar gambar kecil.

API `/api/status` tetap menyertakan sampel langsung `captures` dan `capture_error` pada Kamera 1. Tabel membaca riwayat tersimpan melalui `/api/captures?camera=1&kind=person` dan `kind=no_mask`, dengan respons `{total, people}`: total arsip dan maksimal enam rekaman terbaru. Foto disimpan bersama metadata sebagai BLOB dalam tabel `captures` SQLite agar satu batch tersimpan atomik, lalu dilayani lewat `/api/captures/{id}/image` dan `/portrait`. Riwayat tidak dibersihkan ketika orang keluar, kamera terputus, IP berubah, AI dinonaktifkan, atau aplikasi direstart. Akses foto tetap terautentikasi dan `no-store`. Disk akan terus bertambah karena tidak ada penghapusan otomatis; tentukan izin akses, retensi, kapasitas disk, dan cadangan database sebelum penggunaan nyata.

Tabel tanpa masker menyimpan setiap orang dengan hasil `mask=missing` dari frame Gemini yang sama, bukan mencocokkan ID dengan tangkapan YOLO pada frame lain. Hasil `worn`/`unclear` tidak masuk tabel pelanggaran. Tidak ada permintaan cloud tambahan untuk riwayat atau wajah. Kejadian ringkas lama tetap tersedia di bagian yang dapat dibuka di bawah tabel (enam terbaru), tanpa menghapus database `events` yang sudah ada. Halaman diperbarui setiap 3 detik setelah permintaan sebelumnya selesai; ini polling, bukan push langsung. Pada layar kecil, tabel berada di bawah video dan dapat digulir horizontal. Kamera lain yang dikonfigurasi tetap diproses di server. API `/api/events` tanpa filter tetap mengembalikan maksimal 100 kejadian semua kamera; `?camera=1&kind=no_mask` membatasi ke indikasi tanpa masker Kamera 1. Filter `kind=no_hairnet` juga tersedia.

Penghitung di atas video menampilkan jumlah kotak `person` pada inferensi terbaru Kamera 1, diperbarui di halaman setiap 3 detik. Ini jumlah orang yang sedang terdeteksi, bukan total pengunjung unik atau hitungan masuk/keluar. Angka `0` berarti inferensi tidak menemukan orang; “Tidak tersedia” berarti belum ada hasil, kamera terputus, atau model tidak menyediakan kelas `person`. Hitungan dapat keliru jika orang tertutup, terlalu kecil, atau mendapat kotak ganda. API `/api/status` menyertakan `people_count` per kamera (`null` jika tidak tersedia).

## Menjalankan

Butuh Python 3.9 atau lebih baru, koneksi jaringan ke kamera/XVR dan ruang penyimpanan untuk foto. Dari folder ini:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Isi `.env` dengan `APP_USERNAME` dan `APP_PASSWORD` untuk akun bersama. Gunakan kata sandi yang panjang dan acak; username tidak boleh mengandung `:`. Aplikasi tidak akan mulai jika akun belum diisi.

Isi juga URL RTSP channel yang ingin digunakan. Contoh pola Dahua, jika didukung firmware: `rtsp://USERNAME:PASSWORD@IP:554/cam/realmonitor?channel=1&subtype=1`. Untuk XVR DH-XVR5232AN-S2, sesuaikan `channel` untuk setiap kanal; `subtype=1` umumnya substream dan dapat mengurangi beban, tetapi resolusinya harus cukup jelas untuk melihat masker dan hairnet. Pastikan RTSP diaktifkan, jaringan dapat menjangkau perangkat, dan periksa pola URL pada dokumentasi model/firmware. Encode karakter khusus dalam username/password kamera sebagai URL encoding. Jangan membagikan `.env` atau URL berisi kredensial.

Jika video terlihat pecah saat diperbesar, coba `subtype=0` (main stream) lalu restart server. Resolusi dan bitrate main stream diatur pada perangkat Dahua, bukan oleh ukuran tampilan browser. Main stream meningkatkan beban decoding, jaringan, dan pengiriman JPEG; uji kelancaran pada perangkat yang digunakan. Memperbesar substream di browser tidak menambah detail gambar.

### HTTPS dan akses LAN

Gunakan sertifikat TLS yang dipercaya oleh perangkat pengguna dan mencakup alamat IP atau nama host server dalam Subject Alternative Name (SAN). Simpan sertifikat sebagai `tls-cert.pem` dan kunci privat sebagai `tls-key.pem` di folder aplikasi. Misalnya, untuk uji coba dengan OpenSSL 1.1.1 atau lebih baru, ganti `192.168.1.10` dengan IP LAN server:

```sh
openssl req -x509 -newkey rsa:2048 -nodes -keyout tls-key.pem -out tls-cert.pem \
  -days 365 -subj "/CN=192.168.1.10" -addext "subjectAltName=IP:192.168.1.10"
```

Sertifikat ini ditandatangani sendiri dan tidak otomatis dipercaya browser; administrator harus memasangnya sebagai sertifikat tepercaya pada perangkat pengguna, atau menggunakan sertifikat dari CA yang dipercaya. Jaga kunci privat dan berikan akses hanya kepada administrator. Pastikan OpenSSL mendukung opsi `-addext`.

```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --ssl-keyfile tls-key.pem --ssl-certfile tls-cert.pem
```

Buka `https://192.168.1.10:8000` dengan IP server yang sesuai dari perangkat dalam jaringan lokal, lalu masukkan akun bersama. Aplikasi menolak permintaan HTTP tanpa HTTPS. Batasi port 8000 dengan firewall hanya untuk jaringan pengguna yang tepercaya; jangan lakukan port forwarding ke internet. Perubahan langsung pada `.env` memerlukan restart; gunakan formulir browser di bawah untuk mengganti IP tanpa restart. Untuk menghentikan tekan Ctrl+C. Jika port 8000 sedang digunakan proses lain, pilih port kosong seperti `--port 8001` dan gunakan nomor port tersebut di alamat browser.

### Mengganti IP kamera tanpa restart

Di halaman browser, buka **Ganti IP kamera** di bawah video. Pilih kamera, isi IP baru (IPv4 atau IPv6), lalu klik **Simpan dan sambungkan ulang**. IP disimpan ke `.env` dan hanya koneksi kamera tersebut yang dibuka ulang; model, kamera lain, serta koneksi video browser tetap berjalan. Akun RTSP, port, kanal, dan subtype dipertahankan dan kredensial tidak ditampilkan di formulir/API pengaturan. Pengaturan ini tidak mengubah IP fisik perangkat Dahua; gunakan alamat yang sudah dapat dijangkau server.

Koneksi ulang dapat membutuhkan beberapa detik karena timeout RTSP. Pesan berhasil menyimpan bukan jaminan IP dapat diakses; periksa status kamera. Jika IP salah, simpan kembali alamat yang benar melalui formulir yang sama. Kamera yang belum punya URL RTSP harus dikonfigurasi awal di server. Perubahan akun kamera, kanal, stream, model, dan akun aplikasi lewat `.env` masih memerlukan restart.

Semua pemegang akun bersama dapat mengubah IP. Gunakan hanya satu proses Uvicorn (jangan `--workers` lebih dari satu), agar konfigurasi aktif dan koneksi kamera konsisten. Jika `.env` tidak dapat ditulis, perubahan ditolak dengan pesan kesalahan dan koneksi lama dipertahankan.

## Model deteksi

### Kotak orang sebelum model masker/hairnet tersedia

Model deteksi orang umum yang sudah terlatih dapat dipakai tanpa melatih dataset sendiri. Isi `.env`, lalu restart:

```dotenv
MODEL_PATH=
PERSON_MODEL_PATH=models/yolo11n.pt
AI_CAMERA_ID=1
DETECTION_INTERVAL=0.2
```

Ultralytics dapat mengunduh weights `yolo11n.pt` jika belum tersedia. Mode ini hanya menampilkan kotak `person` hijau, tidak mendeteksi kepatuhan masker/hairnet dan tidak membuat kejadian pelanggaran. Kotak hanya muncul jika orang benar-benar terdeteksi; sudut CCTV, ukuran objek, dan pencahayaan memengaruhi hasil. Jika `MODEL_PATH` diisi, model masker/hairnet tersebut mendapat prioritas; model yang tidak valid menyebabkan startup gagal, bukan diam-diam beralih ke model orang.

### Model masker/hairnet

Sediakan file weights Ultralytics YOLO untuk **object detection** yang dilatih dengan kelas tepat `no_mask` dan `no_hairnet`, lalu isi `MODEL_PATH` dengan path file tersebut. Untuk kotak orang, model juga perlu kelas `person`. Kotak orang berwarna hijau, pelanggaran merah; kotak orang tidak membuat kejadian pelanggaran. Nama kelas harus mendeskripsikan pelanggaran, bukan `mask` atau `hairnet` yang dikenakan. Model umum YOLO tidak otomatis mendeteksi pelanggaran ini. Model perlu dilatih/dievaluasi dengan data yang relevan untuk sudut kamera, pencahayaan, dan jenis hairnet setempat; tinjau hasil secara manual. Status model terlihat di halaman, dan aplikasi menolak model dengan kelas tidak sesuai.

### Gemini 3.5 Flash-Lite (opsional)

Integrasi API disiapkan untuk Kamera 1, tanpa melatih model masker/hairnet sendiri. **Default `GEMINI_ENABLED=0` berarti tidak ada gambar yang dikirim ke cloud.** Untuk konten non-sensitif yang diizinkan, konfigurasi server dapat diisi:

```dotenv
PERSON_MODEL_PATH=models/yolo11n.pt
AI_CAMERA_ID=1
GEMINI_ENABLED=0
GEMINI_API_KEY=YOUR_SERVER_API_KEY
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_INTERVAL=10
```

Simpan API key hanya di `.env` server, jangan di HTML, URL, git, atau chat. Klik **Aktifkan AI** pada halaman, lalu konfirmasi izin pengiriman gambar ke Google. Tidak perlu restart: aktivasi membaca kembali key, nama model, dan interval dari `.env`, kemudian menyimpan `GEMINI_ENABLED=1`. Tombol **Nonaktifkan AI** menyimpan `0`, menghentikan pekerjaan baru, membersihkan hasil pemeriksaan terakhir, dan tidak menghentikan video/kotak/penghitung orang lokal. Permintaan yang sudah dikirim tidak dapat dibatalkan dan mungkin tetap ditagih; hasilnya diabaikan setelah penonaktifan. Aktivasi ulang menunggu worker sebelumnya berhenti agar tidak ada permintaan ganda. Status tersimpan dipakai pada startup berikutnya kecuali ditimpa variabel lingkungan proses.

API terautentikasi `PATCH /api/vision` menerima `{"enabled": true, "consent": true}` untuk aktivasi atau `{"enabled": false}` untuk penonaktifan. Jangan mengaktifkannya pada rekaman sensitif; tinjau izin penggunaan, kebijakan privasi, dan ketentuan penyedia terlebih dahulu. Harga, kuota, dan ketersediaan model mengikuti akun Google; penggunaan dapat menimbulkan biaya. Jika key, model orang, atau kamera yang dibutuhkan belum tersedia, aktivasi ditolak dengan kesalahan eksplisit tanpa mengubah status aktif.

Jika penyimpanan penonaktifan gagal, AI tetap dihentikan pada proses yang berjalan dan halaman menampilkan kesalahan. File `.env` dapat masih bernilai `GEMINI_ENABLED=1`; perbaiki izin tulis dan simpan penonaktifan sebelum restart agar cloud tidak aktif kembali.

#### Mengumpulkan dataset kandidat dari Gemini

Setelah fitur ini dipasang, setiap respons Gemini yang valid menyimpan potongan gambar **setiap orang yang diperiksa**, termasuk hasil `worn`, `missing`, dan `unclear`, ke tabel `dataset_samples` pada database lokal. Label Gemini adalah **saran**, bukan ground truth. Data tersimpan bersama waktu, kamera, nama model, dan skor deteksi orang; tidak ada pengiriman tambahan ke Gemini. Hasil Gemini lama sebelum fitur ini tidak dapat direkonstruksi sebagai dataset lengkap. Sampel berulang dapat memuat orang yang sama dan frame yang mirip.

Buka **Dataset Gemini · tinjau dan ekspor** di dashboard. Tinjau gambar dan pilih sendiri label masker serta hairnet; pilih `Tidak jelas` jika bukti tidak cukup. Sampel yang sudah ditinjau dapat dibuka dan dikoreksi lagi. Gunakan **Unduh sampel ditinjau (.zip)** untuk mengekspor hanya sampel dengan kedua label yang sudah ditinjau. ZIP berisi `images/{id}.jpg` dan `labels.csv` dengan label saran, label tinjauan, waktu tinjauan, waktu pengambilan, kamera, model Gemini, dan skor deteksi orang. Sampel belum ditinjau tidak ikut ekspor. Jika belum ada sampel ditinjau, ekspor ditolak.

Ekspor ini adalah **dataset klasifikasi potongan orang**, bukan dataset YOLO object detection: Gemini tidak memberi kotak masker atau hairnet. Jangan menggunakan potongan orang sebagai kotak pelanggaran tanpa anotasi yang benar; pipeline `train_model.py` di bawah tetap memerlukan gambar dan bounding box YOLO yang diberi label secara terpisah. Untuk mengganti Gemini diperlukan pelatihan, integrasi, dan evaluasi model lokal; dataset saja tidak menghilangkan kebutuhan model AI. Kumpulkan data yang beragam dan seimbang, lakukan pembagian train/validasi/test berdasarkan sesi atau waktu rekaman (bukan mengacak frame berdekatan), dan uji pada video terpisah sebelum digunakan. Hasil `unclear` tidak boleh dianggap patuh atau pelanggaran.

Database dan file ZIP mengandung gambar orang dan dapat terus membesar. Jangan unggah ke repositori publik, tentukan izin dan masa retensi, serta simpan ekspor di lokasi yang aman. Database lokal di `data/` sudah dikecualikan Git; ZIP hasil unduhan harus dikelola terpisah oleh pengguna.

YOLO tetap menangani kotak/penghitung lokal. Worker terpisah hanya mengirim orang dengan skor deteksi **di atas 70%** dari frame sampel; skor tepat `0.70` atau lebih rendah tidak dikirim. Semua orang yang lolos ambang ini disertakan, tanpa syarat wajah terdeteksi. Ambang ini hanya untuk pengiriman Gemini: `CONFIDENCE` tetap mengatur kotak, penghitung, dan riwayat lokal. Ringkasan menyebut jumlah diperiksa dibanding seluruh orang terdeteksi, sehingga orang yang dilewati tidak dianggap sudah diperiksa. Skor orang tinggi tidak menjamin wajah jelas atau hasil masker akurat.

Gambar yang dikirim adalah potongan kotak orang dari frame asli (bukan URL RTSP atau video penuh) melalui HTTPS. Sisi panjang gambar dibatasi 512 piksel tanpa memperbesar gambar kecil. Jeda minimum tetap `GEMINI_INTERVAL` detik dengan satu pekerjaan aktif. Jika tidak ada orang yang lolos ambang, tidak ada gambar disiapkan atau permintaan dikirim, dan jatah interval tidak terpakai. Budget output disesuaikan dengan jumlah gambar hingga batas model; respons terpotong atau tidak lengkap ditolak, bukan dianggap pemeriksaan berhasil. Semakin banyak orang yang lolos, biaya, penggunaan memori, dan waktu analisis meningkat. Ini tidak menjamin semua orang yang lewat di antara sampel diperiksa. Permintaan yang sudah dikirim dapat tetap ditagih meskipun kamera berubah atau server dihentikan.

Hasil untuk masker dan hairnet adalah `worn`, `missing`, atau `unclear`. Hanya `missing` membuat indikasi `no_mask`/`no_hairnet`; `unclear` tidak dianggap pelanggaran. Panel kanan menyimpan riwayat setiap orang dengan `mask=missing` beserta foto orang dan wajah/area kepala dari frame pemeriksaan yang sama. Tabel menampilkan enam terbaru dan tetap ada ketika pemeriksaan terakhir tidak menemukan pelanggaran atau Gemini dinonaktifkan. Ringkasan pemeriksaan terakhir menyebut jumlah hasil tidak jelas dan waktu gambar, bukan status orang pada frame saat ini. ID input Gemini bersifat sementara, berbeda dari ID arsip tangkapan; keduanya bukan identitas atau tracking orang. Tabel lama `events` tetap ringkasan per kamera/jenis sesuai cooldown; tabel baru `captures` tidak menerapkan cooldown per jenis sehingga beberapa orang tanpa masker dalam satu frame semuanya dapat tersimpan. Kotak merah Gemini hanya ada pada foto kejadian ringkas, bukan ditempelkan ke orang lain pada video yang sudah bergerak.

Kejadian menyimpan `source=gemini`. Skor yang ditampilkan untuk kejadian Gemini adalah skor **deteksi orang dari YOLO**, bukan probabilitas kepatuhan masker/hairnet dari Gemini. Kejadian lama tetap disimpan dengan `source=yolo`. Cooldown per jenis pelanggaran/kamera tetap berlaku. Timeout, HTTP 401/429, hasil diblokir, atau JSON tidak valid ditampilkan sebagai kesalahan pemeriksaan, bukan hasil patuh. Uji akurasi pada gambar yang diizinkan dan tinjau hasil petugas sebelum digunakan.

`AI_CAMERA_ID=1` menjalankan inferensi hanya pada Kamera 1; `0` menjalankan pada semua kamera (default jika pengaturan tidak ada). `CONFIDENCE` mengatur ambang kepercayaan (0–1); `DETECTION_INTERVAL` adalah jeda minimum inferensi per kamera dalam detik. Contoh konfigurasi memakai `0.2` detik, tetapi kecepatan nyata tergantung perangkat. Kotak hasil terakhir digambar pada setiap frame sampai inferensi berikutnya, bukan hanya satu frame; kotak dibersihkan jika prediksi berikutnya tidak menemukan orang. Ini deteksi berulang, bukan tracking identitas, dan posisi kotak bisa tertinggal saat orang bergerak cepat. `EVENT_COOLDOWN` adalah jeda minimum penyimpanan per jenis pelanggaran per kamera, bukan per orang. Karenanya beberapa orang atau kejadian berdekatan bisa tercatat sebagai satu kejadian. Inferensi beberapa channel berbagi satu model dan diproses bergantian. Uji beban dan akurasi sebelum digunakan secara operasional.

### Menyiapkan dataset dan pelatihan lokal

Belum ada dataset atau model masker/hairnet terlatih yang disertakan. Skrip tidak mengambil foto CCTV, tidak memberi label otomatis, dan tidak akan melatih jika struktur/label tidak valid. OpenCV menggunakan paket `opencv-python` karena diperlukan Ultralytics; jangan memasangnya bersama `opencv-python-headless` dalam environment yang sama.

Siapkan gambar `.jpg`, `.jpeg`, atau `.png` dan label YOLO dengan struktur:

```text
data/dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Setiap gambar memerlukan file label dengan nama dasar yang sama, misalnya `images/train/frame001.jpg` dan `labels/train/frame001.txt`. Setiap baris label berformat `class_id x_center y_center width height`, dengan koordinat dinormalisasi ke 0–1 terhadap gambar asli:

- `0 person`: kotak seluruh tubuh setiap orang yang terlihat, termasuk yang patuh.
- `1 no_mask`: kotak area wajah yang jelas tidak memakai masker.
- `2 no_hairnet`: kotak area kepala/rambut yang jelas tidak memakai hairnet.

Seseorang tanpa kedua perlengkapan memiliki kotak `person`, `no_mask`, dan `no_hairnet` terpisah. Orang yang memakai perlengkapan dengan benar hanya memiliki kotak `person`; jangan memberi label pelanggaran. Area yang tidak jelas harus ditinjau, bukan diasumsikan melanggar. Sertakan juga gambar kosong/latar dengan file label kosong. Gunakan gambar asli, bukan tangkapan layar video yang diregangkan.

Kedua split harus berisi contoh ketiga kelas. Pisahkan sesi/waktu pengambilan untuk train dan val; jangan menaruh frame yang sama atau berdekatan di kedua split. Sediakan video uji terpisah untuk menilai false positive/false negative pada kondisi CCTV nyata. Pemeriksaan struktur tidak menjamin kualitas label atau akurasi model. Gunakan alat anotasi lokal yang mendukung YOLO; jangan mengunggah rekaman orang ke layanan pihak ketiga tanpa otorisasi.

Periksa dahulu (tidak mengunduh weights atau menjalankan pelatihan):

```sh
.venv/bin/python train_model.py --data dataset.example.yaml --check-only
```

Setelah dataset siap:

```sh
.venv/bin/python train_model.py --data dataset.example.yaml --epochs 50 --device cpu
```

Jika sandbox agen menolak pembuatan semaphore (`multiprocessing.SemLock: Operation not permitted`), pelatihan berhenti meskipun `workers=0`. Tinjau kebijakan sandbox untuk izin tersebut, atau jalankan perintah pelatihan dari terminal lokal Anda. Jangan menganggap hasil tes struktur sebagai bukti bahwa pelatihan selesai.

Default awal memakai `yolo11n.pt` untuk transfer learning; weights umum dapat diunduh oleh Ultralytics bila belum tersedia. Ini bukan model masker/hairnet siap pakai. Untuk weights lokal, gunakan `--model /path/to/local/yolo11n.pt`; Ultralytics mungkin tetap mengunduh aset pendukung seperti font. Sinkronisasi telemetry dan integrasi pencatatan eksternal dinonaktifkan saat memuat model; dataset tetap lokal. Konfigurasi Ultralytics default disimpan di `data/ultralytics/`. Periksa lisensi Ultralytics/weights dan kebutuhan lisensi penggunaan Anda. Pada Apple Silicon, `--device mps` dapat dicoba jika didukung PyTorch; pada NVIDIA dengan CUDA yang sesuai, gunakan `--device 0`. CPU tetap didukung tetapi pelatihan bisa lama. Jangan melatih bersamaan dengan monitor operasional jika sumber daya terbatas.

Skrip mencetak lokasi `best.pt` setelah pelatihan. Evaluasi terlebih dahulu, lalu isi `.env`:

```dotenv
MODEL_PATH=/absolute/path/to/best.pt
AI_CAMERA_ID=1
DETECTION_INTERVAL=0.2
```

Restart server. Status model harus aktif; kotak dan kejadian hanya muncul berdasarkan prediksi model nyata, bukan data contoh.

## Privasi dan keterbatasan

Ini prototipe LAN, bukan sistem siap produksi: akun bersama tidak memiliki audit per pengguna, pembatasan percobaan masuk, atau manajemen sesi. Browser dapat menyimpan akun sampai ditutup; jangan gunakan pada perangkat bersama yang tidak tepercaya. **Jangan buka ke internet.** Foto dan database disimpan lokal di `data/` tanpa penghapusan otomatis. Tentukan akses, masa retensi, pemberitahuan kepada orang yang direkam, dan prosedur penghapusan sebelum penggunaan nyata. Gambar dan hasil AI dapat salah; jangan gunakan sebagai satu-satunya dasar keputusan terhadap seseorang.

Tes unit: `python -m unittest test_detection test_access test_train_model test_gemini_vision`. Tes Gemini menggunakan gambar sintetis dan transport HTTP tiruan; tidak mengirim rekaman atau memverifikasi akurasi layanan cloud.
