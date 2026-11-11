# Soundxx (focus-ear): Project Brief

Dokumen ini konteks untuk perencanaan. Isinya **apa** yang mau dicapai, **seperti apa** hasilnya, **di mana** posisi proyek sekarang, dan **apa** masalahnya. Cara implementasinya sengaja tidak ditulis di sini; itu bagian dari plan.

---

## 1. Konteks

- **Apa ini:** proyek akhir mata kuliah ML / NLP / AI. Dipakai sendiri oleh Jason di kelas kuliah di UC Surabaya.
- **Repo:** `DkXenos/Soundxx`, package Python `focus_ear`. Jalan di Python 3.11.
- **Perangkat target:**
  - MacBook Pro M3 (8 GB RAM), pakai mic bawaan.
  - Galaxy Buds sebagai output. Nama Bluetooth-nya "Mewo".
- **Lingkungan pakai:** ruang kuliah atau hall yang ramai. Ada satu dosen yang paling banyak bicara, sesekali mahasiswa bertanya, dan ada noise ruangan (AC, kipas, kursi, obrolan). Seringnya tidak ada wifi yang bisa diandalkan.
- **Bahasa yang diucapkan:** Indonesia campur istilah teknis Inggris (code-mixed), contohnya "kita pakai loop untuk iterate array". Kadang ada pertanyaan full bahasa Inggris.
- **Referensi pengalaman pakai:** [`jankeesvw/omarchy-meeting-recorder`](https://github.com/jankeesvw/omarchy-meeting-recorder). Ini meeting recorder di Linux.
  - Alurnya: rekam, lalu setelah selesai keluar transkrip per speaker, player yang bisa diputar dari baris mana pun, speaker yang bisa di-rename atau digabung, baris yang bisa diedit atau dipindah ke speaker lain, chapter otomatis, dan folder per sesi.
  - Yang mau diambil adalah **pengalaman pakainya**, bukan platform atau teknologinya.
  - Bedanya dengan Soundxx: app itu tidak live, dibuat untuk meeting online (dua track: mic dan audio komputer), dan jalan di Linux. Soundxx butuh live di telinga, dari satu mic, di ruang kelas, di macOS.

---

## 2. Objektif

Satu app yang bisa dipakai selama satu sesi kuliah penuh (sekitar 90 menit) untuk:

1. **Memisahkan suara orang-orang di ruangan.**
2. **Mengidentifikasi dan memberi label** suara itu milik siapa (misalnya "Pak Daniel", "Mahasiswa 1"), dan tetap dikenali di pertemuan berikutnya.
3. **Mengeraskan speaker yang dipilih dan mengecilkan yang lain**, live, didengar lewat Buds.
4. **Merekam dan mentranskrip** speaker itu, lalu mengubahnya menjadi **catatan kuliah** yang bisa dibaca ulang.

Plus satu objektif non-ML yang sama pentingnya:

5. **Bisa dipakai tanpa harus hafal command.** Dibuka dengan satu aksi, dioperasikan lewat UI yang jelas (bukan TUI terminal), dan memberi tahu dengan bahasa manusia kalau ada yang salah.

Karena ini proyek mata kuliah, komponen ML-nya harus **bisa dijelaskan dan diukur** untuk laporan dan demo. Artinya ada angka evaluasi dari rekaman nyata, bukan cuma "kelihatannya jalan".

---

## 3. Expected outcome

### 3.1 Alur pakai yang diharapkan

**Sebelum kelas**
- App dibuka dengan satu aksi. Tidak perlu aktifkan venv, `cd`, atau mengetik flag.
- Layar awal langsung menunjukkan:
  - mic yang dipakai, dengan meter level yang bergerak;
  - output yang dipakai (Buds terdeteksi atau tidak).
- Output dipilih dari daftar perangkat yang ada, tidak hardcoded ke satu nama. Kalau Buds tidak tersambung, app bilang begitu dengan jelas, tidak diam menunggu.
- Kalau ada yang menghalangi (izin mic belum diberikan, model belum terunduh, output tidak aman), app menjelaskan masalahnya dan langkah perbaikannya **sebelum** sesi dimulai.
- Bisa mengisi nama sesi atau mata kuliah (opsional), lalu tekan **Start**.

**Selama kelas**
- Suara ruangan terdengar di Buds, dengan speaker terpilih lebih keras dan yang lain lebih pelan.
- Daftar speaker muncul otomatis begitu mereka bicara. Setiap speaker punya:
  - label;
  - indikator sedang bicara;
  - level suara;
  - total durasi bicara.
- Memilih, mengganti nama, dan mereset speaker bisa dilakukan dengan klik.
- Speaker yang sudah pernah dinamai di sesi sebelumnya dikenali dan dipilih otomatis.
- Transkrip live terlihat berjalan, dengan label speaker.
- Status yang selalu terlihat:
  - indikator rekaman;
  - durasi sesi;
  - status Buds (tersambung, terputus, atau sedang reconnect);
  - peringatan penting seperti mic clipping, mic sunyi, atau transkripsi tertinggal.
- Ada **Pause/Resume** dan **Stop**.
- App tetap stabil selama 90 menit, termasuk saat Buds putus-sambung.

**Setelah kelas**
- Setiap sesi tersimpan sebagai satu unit (misalnya satu folder per sesi) berisi:
  - **audio mentah** seluruh sesi;
  - **transkrip final** yang kualitasnya lebih baik dari transkrip live;
  - **catatan kuliah**: ringkasan, poin penting, dan pembagian per topik/chapter dengan timestamp;
  - metadata sesi: tanggal, mata kuliah, speaker, durasi, model yang dipakai.
- Hasil sesi bisa dibuka lagi kapan saja di app:
  - Transkrip dan catatan tampil berdampingan.
  - Ada player audio; klik satu baris transkrip memutar audio dari titik itu.
  - Label speaker bisa dikoreksi: rename, gabung dua speaker jadi satu, pindahkan satu baris ke speaker lain.
  - Teks transkrip bisa diedit langsung.
- Ada daftar sesi lama, bisa difilter per mata kuliah.
- Sesi yang terputus di tengah (crash, laptop mati, app ditutup paksa) tidak hilang. Rekamannya bisa dipulihkan saat app dibuka lagi.

### 3.2 Kriteria per objektif

| # | Objektif | Dianggap tercapai kalau |
|---|---|---|
| 1 | Pisahkan suara | Saat giliran bicara bergantian, setiap orang diperlakukan sebagai speaker terpisah. Saat dua orang bicara **bersamaan**, speaker terpilih tetap terdengar lebih jelas dari yang lain. Bagian yang overlap ini diperlakukan sebagai **eksperimen terukur** (dibandingkan dengan kondisi tanpa pemisahan), bukan syarat app bisa dipakai. |
| 2 | Identifikasi + label | Di rekaman kuliah nyata, dosen tidak terpecah menjadi beberapa speaker dan tidak tergabung dengan mahasiswa. Nama yang diberikan bertahan ke sesi berikutnya. Label yang salah bisa dibetulkan setelah kelas. |
| 3 | Keraskan/kecilkan | Speaker terpilih terdengar penuh, yang lain turun jelas (default −20 dB). Tidak ada klik atau letupan saat pergantian. Latency di telinga tidak lebih buruk dari sekarang (sekitar 650 ms). |
| 4 | Rekam + catatan | Audio mentah selalu tersimpan. Transkrip final terbaca benar untuk ucapan dosen dalam bahasa campur Indonesia–Inggris. Catatan berisi ringkasan dan poin per topik yang sesuai isi kuliah. |
| 5 | Mudah dipakai | Orang yang belum pernah melihat repo ini bisa memulai sesi tanpa membaca README dan tanpa mengetik command. Semua kegagalan umum (Buds tidak ada, izin mic, model belum diunduh, tidak ada internet) muncul sebagai pesan yang jelas di UI. |

### 3.3 Hasil untuk laporan mata kuliah

- Minimal satu (lebih baik beberapa) **rekaman kuliah nyata** yang dipakai sebagai data evaluasi.
- Angka evaluasi yang bisa dilaporkan, misalnya:
  - akurasi label speaker;
  - jumlah speaker yang ditemukan dibanding jumlah sebenarnya;
  - kualitas transkrip live dibanding transkrip final;
  - efek noise suppression;
  - latency;
  - pemakaian memori selama 90 menit.
- Demo yang bisa ditunjukkan langsung.

---

## 4. Kondisi sekarang (per 11 Nov 2026)

### 4.1 Status repo

- **Commit terakhir:** 27 Sep 2026, "sound major installation with new library and sdk". Commit ini berisi sekitar 3.400 baris perubahan sekaligus.
- **Commit sebelumnya:** "version 2 checkopint reached untested".
- Total 7 commit.
- Belum pernah diuji di ruang kelas nyata. Semua angka di README berasal dari **suara sintetis** (TTS macOS) dan simulasi reverb/noise ruangan.

### 4.2 Yang sudah ada di kode

Pipeline live (`python -m focus_ear`):

| Tahap | Isi |
|---|---|
| Input | High-pass 80 Hz, lalu noise suppression DeepFilterNet3 versi streaming. Default-nya hanya untuk audio yang didengar (`--denoise-scope playback`). |
| 1. Deteksi bicara | silero-vad lewat ONNX Runtime |
| 2. Noise gate | Mengecilkan saat tidak ada yang bicara (−18 dB) |
| 3. Identifikasi speaker | Voiceprint ECAPA (speechbrain, jalan di GPU Apple). Clustering online dengan window 1.5 s, threshold 0.45, merge 0.55, maksimal 8 speaker. Speaker yang tidak dinamai dilupakan setelah 120 s. |
| 4. Gain per speaker | Speaker terpilih 0 dB, yang lain −20 dB. Lookahead 250 ms, ramp 50 ms. |
| 5. Transkripsi | Whisper small (MLX, GPU Apple), hanya speaker terpilih. Bahasa default `id` dengan prompt kalimat code-mixed. Fallback ke faster-whisper di CPU. |
| Output | AGC (target −23 dBFS) dan limiter (ceiling −6 dBFS) |

Fitur lain yang sudah ada:
- **Ketahanan sesi panjang:** reconnect Buds, deteksi clipping, memori terbatas (sekitar 3.1 GB stabil di soak test 45 menit), Ctrl-C yang bersih, dan health log.
- **UI:** TUI di terminal (Textual).
  - Tombol: `1`–`9` pilih speaker, `0` semua speaker, `n` beri nama, `r` reset, `t` transkrip, `d` debug, `q` keluar.
- **Profil speaker:** nama tersimpan di `~/.focus-ear/speakers.json`. Speaker dikenali dan dipilih otomatis di sesi berikutnya.
- **Output sesi:** `~/.focus-ear/sessions/<tanggal_jam>.md` (transkrip yang terbaca) dan `.jsonl` (per ucapan, berisi skor kualitas Whisper untuk evaluasi).
- **Mode file:** `--file rekaman.wav` menjalankan rantai yang sama atas rekaman yang sudah ada (VAD, speaker, Whisper) dan menghasilkan file yang sama.
- **Extension point** untuk pemisahan suara overlap ada di `pipeline.py`, tapi belum diimplementasi.

### 4.3 Cara menjalankan sekarang

```sh
cd ~/Documents/PYTHON/Programs/Soundxx
python3.11 -m venv venv && source venv/bin/activate          # sekali
pip install -r requirements.txt                              # sekali
pip install --no-deps -r requirements-denoise.txt            # sekali
python -m focus_ear --list-devices                           # cek perangkat
python -m focus_ear                                          # jalan
```

Selain itu ada sekitar 45 flag CLI untuk tuning.

Run pertama mengunduh sekitar 570 MB model:
- speaker model sekitar 80 MB;
- Whisper small sekitar 480 MB;
- DeepFilterNet sekitar 8 MB.

### 4.4 Hasil test

`python -m unittest discover -s tests` dijalankan di Linux (bukan Mac): 86 test.
- **72 lulus.**
- **8 di-skip** karena butuh model yang belum diunduh.
- **6 error**, semuanya karena PortAudio tidak tersedia di environment itu, bukan bug logika.
- Logika VAD, gain, clustering, transkrip, denoise, dan DSP lolos.
- Test yang butuh perangkat audio asli dan model harus dijalankan ulang di Mac.

---

## 5. Masalah

### P0: App tidak bisa dipakai oleh pemiliknya sendiri

- Terakhir kali dicoba di Mac, **tidak ada fitur yang terasa jalan**. Penyebab pastinya belum diketahui, karena belum ada log atau output dari run yang gagal.
- Dugaan penyebab dari pembacaan kode, belum diverifikasi:
  - Output device di-hardcode `"Mewo"` (`config.py`). Kalau Buds tidak tersambung, app diam menunggu (`waiting for 'Mewo'`) dan tidak ada suara.
  - Output ke speaker MacBook sengaja ditolak (untuk mencegah feedback), tanpa alternatif yang jelas bagi pengguna.
  - Izin mic belum diberikan ke Terminal. macOS lalu memberi audio sunyi tanpa error. App memberi peringatan, tapi peringatannya mudah terlewat.
  - Run pertama mengunduh sekitar 570 MB model tanpa umpan balik yang jelas, jadi terasa seperti hang. Tanpa internet, run pertama gagal.
  - Versi terakhir mungkin belum pernah dijalankan utuh setelah commit besar 27 Sep.
- Masalah usability:
  - UI di terminal membingungkan untuk dinavigasi.
  - Ada banyak command dan flag, dan tidak jelas mana yang wajib.
  - Setup venv manual.
  - Tidak ada konsep Start/Stop sesi; app langsung jalan begitu dibuka.

### P0: Audio mentah tidak direkam

- Mode live tidak menyimpan audio sama sekali; yang tersimpan hanya transkrip.
- Akibatnya:
  - Kuliah tidak bisa diproses ulang dengan kualitas lebih baik.
  - Tidak bisa diputar ulang.
  - Label speaker tidak bisa dikoreksi terhadap audio.
  - Tidak ada data nyata untuk evaluasi dan laporan.

### P1: Kualitas label dan transkrip terbatas karena semuanya live

- Identifikasi speaker hanya berupa clustering online. Keputusannya diambil dalam sekitar 1 detik, tanpa melihat sisa rekaman.
- Transkripsi hanya memakai Whisper small, karena harus muat dan cepat selama live.
- Tidak ada tahap "setelah kelas" yang memperbaiki hasil live.
- Label yang salah tidak bisa dikoreksi setelah sesi selesai.

### P1: Belum ada catatan

- Output-nya hanya transkrip mentah. Belum ada ringkasan, poin penting, atau pembagian topik/chapter.
- Belum ada tempat untuk membaca atau mengelola hasil sesi: tidak ada daftar sesi, tidak ada player, tidak ada edit.

### P1: Pemisahan suara overlap belum ada

- Yang ada sekarang hanya mengatur volume per giliran bicara. Saat dua orang bicara bersamaan, keduanya mendapat volume yang sama.
- Identifikasi juga butuh sekitar 1 detik. Akibatnya, sekitar 1 detik pertama setiap pembicara baru lolos dengan volume penuh, dan interupsi tanpa jeda memakai volume speaker sebelumnya selama sekitar 1 detik.
- Ini bagian yang paling berisiko secara teknis: real-time, satu mic, dan laptop 8 GB.

### P2: Belum terbukti di dunia nyata

- Semua parameter (threshold clustering, window, gate, denoise scope, prompt Whisper) dituning di data sintetis.
- Belum diketahui:
  - apakah mic MacBook menangkap dosen dengan cukup jelas dari posisi duduk;
  - apakah satu dosen terpecah jadi beberapa speaker di ruangan nyata;
  - apakah denoise membantu atau justru merusak.

---

## 6. Batasan

- **Hardware:** MacBook Pro M3, **8 GB RAM**. Pipeline live sudah memakai sekitar 3.1 GB. Apa pun yang ditambahkan, live maupun setelah kelas, harus muat.
- **Offline:** di ruang kelas sering tidak ada wifi. Semua yang dibutuhkan selama sesi harus sudah ada di laptop. Unduhan hanya boleh terjadi saat setup.
- **Tanpa budget API berbayar.** Tidak boleh bergantung pada API AI berbayar. Yang tersedia: model lokal, dan Claude Code yang sudah login (langganan) di Mac.
- **Platform:** macOS Apple Silicon saja. Tidak perlu Windows, Linux, atau mobile.
- **Data lokal:** rekaman berisi suara dosen dan mahasiswa lain. Semua tetap di laptop dan tidak diunggah ke mana pun.
- **Jangan merusak yang sudah jalan:**
  - aturan audio callback yang sekarang (hanya copy, tanpa inference, alokasi, lock, atau log di callback);
  - latency;
  - ketahanan sesi panjang;
  - test yang ada.
- **Proyek kuliah:** harus bisa dijelaskan, didemokan, dan diukur.

---

## 7. Keputusan yang masih terbuka

Belum diputuskan, perlu dijawab saat perencanaan:

- **Bentuk UI:** web lokal di browser, app macOS native, atau app menu bar. Yang sudah pasti: bukan TUI terminal sebagai UI utama.
- **Status pemisahan suara overlap:** fitur wajib untuk versi final, atau eksperimen terpisah untuk laporan.
- **Format dan lokasi catatan:** Markdown di folder sesi saja, atau juga diekspor ke tempat lain.
- **Batas waktu proyek:** deadline pengumpulan dan demo mata kuliah belum dicatat di sini. _(isi: ____)_
- **Mahasiswa lain di transkrip:** apakah mereka ikut ditranskrip dan dicatat, atau hanya dosen. Ini menyangkut privasi dan izin merekam di kelas.

---

## 8. Di luar scope

- Meeting online atau rekaman audio sistem (Zoom, Meet).
- Banyak pengguna, sinkronisasi cloud, atau berbagi sesi.
- Platform selain macOS Apple Silicon.
- Penerjemahan.
