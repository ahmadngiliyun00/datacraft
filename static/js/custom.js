document.addEventListener('DOMContentLoaded', () => {
	const cards = document.querySelectorAll('.card.step');
	const collapses = document.querySelectorAll('.collapse');
	const uploadForm = document.getElementById('upload-form');
	const uploadButton = document.getElementById('upload-button');
	const uploadText = document.getElementById('upload-text');
	const uploadSpinner = document.getElementById('upload-spinner');

	const collapseA = document.getElementById('collapseA'); // Unggah Dataset
	const collapseB = document.getElementById('collapseB'); // Rincian Dataset

	const uploadCard = document.getElementById('cardA');
	const detailsCard = document.getElementById('cardB');

	const fileInput = document.getElementById('fileInput');
	const submitBtn = document.getElementById('submit-btn');

	// Jika dataset sudah diunggah, perbarui UI
	if (datasetUploaded) {
		uploadCard.classList.remove('danger');
		uploadCard.classList.add('success');
		uploadCard.querySelector('.avatar').classList.remove('danger');
		uploadCard.querySelector('.avatar').classList.add('success');
		uploadCard.querySelector('.avatar i').classList.remove('bx-x-circle', 'danger');
		uploadCard.querySelector('.avatar i').classList.add('bx-check-circle', 'success');
		uploadCard.querySelector('.badge').classList.remove('bg-danger');
		uploadCard.querySelector('.badge').classList.add('bg-success');
		uploadCard.querySelector('.badge').textContent = "Sudah dilakukan";

		// Tampilkan collapse unggah dataset secara default
		let bsCollapseB = new bootstrap.Collapse(collapseB, { toggle: false });
		bsCollapseB.show();

		fileInput.disabled = true;
		submitBtn.disabled = true;
		
		// Perbarui status rincian dataset
		detailsCard.classList.remove('danger');
		detailsCard.classList.add('success');
		detailsCard.querySelector('.avatar').classList.remove('danger');
		detailsCard.querySelector('.avatar').classList.add('success');
		detailsCard.querySelector('.avatar i').classList.remove('bx-x-circle', 'danger');
		detailsCard.querySelector('.avatar i').classList.add('bx-check-circle', 'success');
		detailsCard.querySelector('.badge').classList.remove('bg-danger');
		detailsCard.querySelector('.badge').classList.add('bg-success');
		detailsCard.querySelector('.badge').textContent = "Dataset tersedia";
	} else {
		// Tampilkan collapse unggah dataset secara default
		let bsCollapseA = new bootstrap.Collapse(collapseA, { toggle: false });
		bsCollapseA.show();
	}

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
