function showLoading(event) {
	// Prevent default submit behavior
	event.preventDefault();

	// Ambil elemen tombol dan spinner
	const buttonText = document.getElementById('button-text');
	const spinner = document.getElementById('loading-spinner');
	const submitBtn = document.getElementById('submit-btn');

	// Tampilkan spinner dan ubah teks tombol
	buttonText.textContent = 'Memproses...';
	spinner.classList.remove('d-none');

	// Nonaktifkan tombol agar tidak bisa diklik lagi
	submitBtn.disabled = true;

	// Submit form secara manual
	event.target.closest('form').submit();
}
document.addEventListener('DOMContentLoaded', () => {
	const cards = document.querySelectorAll('.card.step');
	const collapses = document.querySelectorAll('.collapse');

	cards.forEach((card) => {
		// Add Bootstrap shadow class by default
		card.classList.add('shadow-sm');

		card.addEventListener('mouseenter', () => {
			card.style.transform = 'translateY(-10px)';
			// Increase shadow on hover
			card.classList.remove('shadow-sm');
			card.classList.add('shadow-lg');
		});

		card.addEventListener('mouseleave', () => {
			card.style.transform = 'translateY(0)';
			// Restore original shadow
			card.classList.remove('shadow-lg');
			card.classList.add('shadow-sm');
		});
	});
	// Close other collapses when one is opened
	collapses.forEach((collapse) => {
		collapse.addEventListener('show.bs.collapse', (e) => {
			collapses.forEach((other) => {
				if (other !== e.target) {
					bootstrap.Collapse.getInstance(other)?.hide();
				}
			});
		});
	});
});
