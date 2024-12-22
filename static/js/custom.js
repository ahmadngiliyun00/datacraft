function showLoading(event) {
  // Prevent default submit behavior
  event.preventDefault();

  // Ambil elemen tombol dan spinner
  const buttonText = document.getElementById("button-text");
  const spinner = document.getElementById("loading-spinner");
  const submitBtn = document.getElementById("submit-btn");

  // Tampilkan spinner dan ubah teks tombol
  buttonText.textContent = "Memproses...";
  spinner.classList.remove("d-none");

  // Nonaktifkan tombol agar tidak bisa diklik lagi
  submitBtn.disabled = true;

  // Submit form secara manual
  event.target.closest('form').submit();
}