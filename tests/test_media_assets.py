"""Reference window, metadata stripping, the one-upload-per-file rule and the shared library."""
import base64
from io import BytesIO
import sqlite3
import unittest

from media_support import MediaCase, noisy_jpeg, small_jpeg
from media import assets, db
from PIL import Image


class WindowTests(MediaCase):
    def refused(self, data, pattern):
        with self.assertRaisesRegex(assets.AssetRejected, pattern):
            assets.check_window(data)

    def test_a_normal_photo_passes(self):
        self.assertEqual(assets.check_window(noisy_jpeg(800, 1200)), (800, 1200))

    def test_too_small_in_bytes(self):
        self.refused(small_jpeg(), "minimum is 20 KB")

    def test_too_large_in_bytes(self):
        self.refused(b"x" * (assets.MAX_BYTES + 1), "maximum is 10 MB")

    def test_short_side(self):
        self.refused(noisy_jpeg(400, 800), "at least 512")

    def test_too_narrow(self):
        self.refused(noisy_jpeg(600, 1600), "too narrow")

    def test_not_an_image(self):
        self.refused(b"not an image" * 4000, "readable image")

    def test_gif_is_refused(self):
        buf = BytesIO()
        Image.effect_noise((800, 800), 64).save(buf, format="GIF")
        self.refused(buf.getvalue(), "isn't accepted")


class CleaningTests(MediaCase):
    def exif_photo(self):
        image = Image.effect_noise((800, 1200), 64).convert("RGB")
        exif = Image.Exif()
        exif[0x0112] = 6                # rotate 90 degrees on display
        exif[0x010F] = "Phone maker"
        gps = exif.get_ifd(0x8825)
        gps[2] = (37.0, 46.0, 0.0)
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=92, exif=exif)
        return buf.getvalue()

    def test_gps_and_camera_data_are_gone_and_the_photo_is_upright(self):
        asset, existed = assets.store(self.exif_photo())
        self.assertFalse(existed)
        stored = Image.open(asset.path)
        self.assertEqual(stored.size, (1200, 800))      # the EXIF rotation was applied
        self.assertEqual(dict(stored.getexif()), {})
        self.assertNotIn(b"Phone maker", asset.read())

    def test_big_photos_are_stored_at_2048(self):
        asset, _ = assets.store(noisy_jpeg(3000, 2400, quality=75))
        self.assertEqual(max(Image.open(asset.path).size), 2048)


class StoreTests(MediaCase):
    def test_same_file_is_stored_once(self):
        data = noisy_jpeg()
        first, existed = assets.store(data)
        second, existed_again = assets.store(data)
        self.assertEqual(first.id, second.id)
        self.assertFalse(existed)
        self.assertTrue(existed_again)
        self.assertEqual(first.id, assets.asset_id(data))

    def test_frames_skip_the_window_and_a_reference_upload_keeps_them(self):
        data = noisy_jpeg(600, 600)
        frame, _ = assets.store(small_jpeg(), "frame")
        self.assertEqual(db.asset(frame.id)["purpose"], "frame")
        asset, _ = assets.store(data, "frame")
        assets.store(data, "reference")
        self.assertEqual(db.asset(asset.id)["purpose"], "reference")

    def test_sweep_removes_old_frames_and_keeps_references(self):
        frame, _ = assets.store(small_jpeg(), "frame")
        reference, _ = assets.store(noisy_jpeg())
        self.assertEqual(assets.sweep_frames(before=2**40), 1)
        self.assertFalse(assets.exists(frame.id))
        self.assertFalse(frame.path.exists())
        self.assertTrue(assets.exists(reference.id))

    def test_a_missing_reference_says_to_upload_again(self):
        with self.assertRaisesRegex(KeyError, "Upload it again"):
            assets.load("0" * 32)


class LibraryTests(MediaCase):
    """Every reference uploaded for anyone can be picked again without uploading it."""

    def test_references_are_listed_newest_first_with_a_thumbnail_and_frames_are_not(self):
        old, _ = assets.store(noisy_jpeg(800, 1200), "reference", "Maya, Face 1")
        db._run("UPDATE assets SET created_at = 1 WHERE id = ?", (old.id,))
        new, _ = assets.store(noisy_jpeg(1200, 800), "reference", "  Persona,\n Body ")
        assets.store(noisy_jpeg(600, 600), "frame")
        page = assets.library()
        self.assertEqual([a["id"] for a in page["assets"]], [new.id, old.id])
        self.assertFalse(page["more"])
        first = page["assets"][0]
        self.assertEqual((first["label"], first["width"], first["height"]), ("Persona, Body", 1200, 800))
        thumb = Image.open(BytesIO(base64.b64decode(first["thumb"].split(",", 1)[1])))
        self.assertEqual(max(thumb.size), assets.THUMB_EDGE)

    def test_pages(self):
        for _ in range(3):
            assets.store(noisy_jpeg())
        page = assets.library(limit=2)
        self.assertEqual((len(page["assets"]), page["more"]), (2, True))
        rest = assets.library(limit=2, offset=2)
        self.assertEqual((len(rest["assets"]), rest["more"]), (1, False))

    def test_removing_only_hides_it_and_uploading_it_again_brings_it_back(self):
        data = noisy_jpeg()
        asset, _ = assets.store(data, "reference", "Maya, Face 1")
        self.assertTrue(assets.hide(asset.id))
        self.assertEqual(assets.library()["assets"], [])
        self.assertTrue(assets.exists(asset.id), "a slot still using it keeps working")
        assets.store(data, "reference", "Someone else, Body")
        listed = assets.library()["assets"]
        self.assertEqual([(a["id"], a["label"]) for a in listed], [(asset.id, "Maya, Face 1")], "the first label stays")
        self.assertFalse(assets.hide("0" * 32))

    def test_a_frame_uploaded_again_as_a_reference_joins_with_its_label(self):
        data = noisy_jpeg(600, 600)
        frame, _ = assets.store(data, "frame", "not kept for frames")
        self.assertIsNone(db.asset(frame.id)["label"])
        self.assertEqual(assets.library()["assets"], [])
        assets.store(data, "reference", "Maya, Face 2")
        self.assertEqual([a["label"] for a in assets.library()["assets"]], ["Maya, Face 2"])

    def test_an_older_database_gets_the_new_columns(self):
        path = self.data / "old.db"
        old = sqlite3.connect(str(path))
        old.execute("CREATE TABLE assets (id TEXT PRIMARY KEY, path TEXT NOT NULL, mime TEXT NOT NULL, "
                    "bytes INTEGER NOT NULL, purpose TEXT NOT NULL DEFAULT 'reference', created_at INTEGER NOT NULL)")
        old.execute("INSERT INTO assets VALUES ('a', '/gone.jpg', 'image/jpeg', 1, 'reference', 1)")
        old.commit()
        old.close()
        db.connect(path)
        row = db.asset("a")
        self.assertEqual((row["label"], row["hidden"]), (None, 0))
        self.assertEqual(assets.library()["assets"], [], "a missing file is skipped, not an error")


class FitTests(MediaCase):
    def test_refs_shrink_until_the_whole_request_fits(self):
        batch = [assets.store(noisy_jpeg(1800, 2400, quality=85))[0] for _ in range(1)] * 10
        images = assets.fit_refs(batch, 4_400_000)
        self.assertLessEqual(sum(len(i) * 4 // 3 for i in images), 4_400_000)
        self.assertEqual(len(images), 10)

    def test_impossible_requests_fail_instead_of_being_cut(self):
        asset, _ = assets.store(noisy_jpeg(1800, 2400, quality=85))
        with self.assertRaisesRegex(RuntimeError, "too large"):
            assets.fit_refs([asset] * 10, 100_000)


if __name__ == "__main__":
    unittest.main()
