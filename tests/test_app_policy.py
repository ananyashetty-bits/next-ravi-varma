"""Content policy (app/policy.py): ~20 allowed and ~20 blocked examples, including borderline ones."""

import pytest

from app.policy import check, check_all

ALLOWED = [
    "Sita sitting alone under a tree in the Ashoka grove, sorrowful, the garden quiet at dusk",
    "Hanuman kneeling before Rama and offering him Sita's ring",
    "Krishna playing the flute under a kadamba tree while cows graze around him",
    "Arjuna aiming his bow at the fish target at Draupadi's swayamvara",
    "Bhishma lying on a bed of arrows on the battlefield of Kurukshetra while the Pandavas gather",
    "Karna struggling to lift his chariot wheel from the mud while Arjuna aims an arrow at him",
    "a fierce battle between the Pandava and Kaurava armies, chariots and elephants charging",
    "Durga riding a lion and fighting the buffalo demon Mahishasura",
    "Kali standing on Shiva, her tongue out, in the cremation ground",
    "Vamana placing his foot on the head of King Bali",
    "Draupadi's disrobing in the royal court, Krishna's endless sari protecting her",
    "Shakuntala writing a love letter on a lotus leaf",
    "Damayanti in the forest, sad, searching for Nala",
    "Ravana carrying Sita away in his flying chariot",
    "Yashoda scolding little Krishna for stealing butter",
    "a hot afternoon on the banks of the Yamuna, Radha waiting for Krishna",
    "Lakshmi seated on a lotus, gold coins falling from her hand",
    "Draupadi humiliated in the dice hall while the elders look away",
    "an ugly demon attacking the sages in the forest",
    "a queen in a red saree on a palace balcony at sunset",
    "",
]

BLOCKED = [
    ("a nude woman bathing in the river", "sexual"),
    ("naked Krishna with the gopis", "sexual"),
    ("Parvati in lingerie", "sexual"),
    ("Draupadi's disrobing, naked in the court", "sexual"),
    ("sexy Lakshmi posing", "sexual"),
    ("Radha and Krishna kissing passionately in bed", "sexual"),
    ("Ganesha as a cartoon eating a burger", "demeaning"),
    ("Hanuman drunk and dancing in a nightclub", "demeaning"),
    ("a funny meme of Shiva", "demeaning"),
    ("Rama as a clown with a red nose", "demeaning"),
    ("Krishna holding a gun", "demeaning"),
    ("a caricature of Vishnu with a huge belly", "demeaning"),
    ("a swastika flag over the temple", "hateful"),
    ("Arjuna's enemies disembowelled on the battlefield", "gore"),
    ("Bhima tearing out Dushasana's entrails, blood everywhere", "gore"),
    ("a severed head held up by the warrior", "gore"),
    ("Narendra Modi as Rama", "living_people"),
    ("Shah Rukh Khan playing Arjuna", "living_people"),
    ("Virat Kohli as Karna holding a bow", "living_people"),
    ("Elon Musk in a royal court", "living_people"),
]


@pytest.mark.parametrize("text", ALLOWED)
def test_allowed(text):
    r = check(text)
    assert r["allowed"], (text, r)


@pytest.mark.parametrize("text,category", BLOCKED)
def test_blocked(text, category):
    r = check(text)
    assert not r["allowed"] and r["category"] == category, (text, r)
    assert r["message"] and "\n" not in r["message"]


def test_near_figure_terms_need_a_figure_nearby():
    assert check("a cartoon of a mountain landscape")["allowed"]
    assert not check("Ganesha as a cartoon")["allowed"]
    far = "Ganesha " + " ".join(["word"] * 12) + " cartoon"
    assert check(far)["allowed"]


def test_case_accents_and_hyphens():
    assert not check("NUDE figure")["allowed"]
    assert not check("Sita, nudé")["allowed"]
    assert not check("pool-of-blood on the field")["allowed"]


def test_word_boundaries():
    assert check("a trumpet player in the procession")["allowed"]   # "trump"
    assert check("Vishnu's vahana Garuda in flight")["allowed"]     # "hot" / "nude" inside other words


def test_check_all_checks_every_text():
    assert not check_all("Hanuman in the forest", "rrvarma style, Hanuman naked, oil painting")["allowed"]
    assert check_all("Hanuman in the forest", "rrvarma style, Hanuman in the forest, oil painting")["allowed"]


def test_negative_prompt_words_are_not_policy():
    # the app checks only the user text and the positive prompt; M3's negative contains "ugly" on purpose
    assert check("an ugly demon attacking the sages")["allowed"]
