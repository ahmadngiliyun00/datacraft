document.addEventListener('DOMContentLoaded', () => {
	const cards = document.querySelectorAll('.card.step');
	const collapses = document.querySelectorAll('.collapse');
	const uploadForm = document.getElementById('upload-form');
	const uploadButton = document.getElementById('upload-button');
	const uploadText = document.getElementById('upload-text');
	const uploadSpinner = document.getElementById('upload-spinner');

	// Efek Hover pada Card
	cards.forEach((card) => {
		card.classList.add('shadow-sm'); // Tambahkan efek shadow default

		card.addEventListener('mouseenter', () => {
			card.style.transform = 'translateY(-5px)';
			card.classList.remove('shadow-sm');
			card.classList.add('shadow-lg');
		});

		card.addEventListener('mouseleave', () => {
			card.style.transform = 'translateY(0)';
			card.classList.remove('shadow-lg');
			card.classList.add('shadow-sm');
		});

		// Toggle Collapse Saat Card Diklik
		card.addEventListener('click', () => {
			let targetCollapse = document.querySelector(card.dataset.bsTarget);

			// Tutup semua collapse sebelum membuka yang baru
			collapses.forEach((collapse) => {
				if (collapse !== targetCollapse) {
					let bsCollapse = bootstrap.Collapse.getInstance(collapse);
					if (bsCollapse) {
						bsCollapse.hide();
					}
				}
			});

			// Tampilkan collapse yang diklik
			let bsTargetCollapse = bootstrap.Collapse.getInstance(targetCollapse);
			if (!bsTargetCollapse) {
				bsTargetCollapse = new bootstrap.Collapse(targetCollapse);
			}
			bsTargetCollapse.show();
		});
	});

	// Event Listener untuk Unggah Form
	if (uploadForm) {
		uploadForm.addEventListener('submit', (event) => {
			event.preventDefault(); // Hindari pengiriman form langsung

			// Tampilkan Loading Indicator
			uploadText.textContent = 'Mengunggah...';
			uploadSpinner.classList.remove('d-none');
			uploadButton.disabled = true;

			// Simulasi Unggah (Opsional: Ganti dengan AJAX jika perlu)
			setTimeout(() => {
				uploadForm.submit(); // Kirim Form setelah simulasi loading
			}, 1000);
		});
	}

	// Pastikan hanya satu collapse terbuka saat halaman dimuat
	collapses.forEach((collapse) => {
		collapse.addEventListener('show.bs.collapse', (event) => {
			collapses.forEach((other) => {
				if (other !== event.target) {
					let bsOtherCollapse = bootstrap.Collapse.getInstance(other);
					if (bsOtherCollapse) {
						bsOtherCollapse.hide();
					}
				}
			});
		});
	});
});
