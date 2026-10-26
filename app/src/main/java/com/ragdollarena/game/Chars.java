package com.ragdollarena.game;

/**
 * The cast of "типичные человечки". Every fighter is drawn procedurally, so a
 * character is just a bag of numbers and colours.
 */
public final class Chars {

    public static final int PERK_NONE = 0;
    public static final int PERK_DASH = 1;    // extra damage while dashing
    public static final int PERK_CD = 2;      // faster dash recharge
    public static final int PERK_ARMOR = 3;   // damage reduction
    public static final int PERK_SYNERGY = 4; // team wide power bonus
    public static final int PERK_REGEN = 5;   // heals the team after a fight

    public static class Data {
        public final String id;
        public final String name;
        public final String role;
        public final String perkName;
        public final String perkDesc;
        public final float hp;
        public final float power;
        public final float mass;
        public final float speed;
        public final float dashCd;
        public final int perk;
        public final int skin, hair, top, bottom, accent, shoe;
        public final int hairStyle, accessory, body;

        Data(String id, String name, String role, String perkName, String perkDesc,
             float hp, float power, float mass, float speed, float dashCd, int perk,
             int skin, int hair, int top, int bottom, int accent, int shoe,
             int hairStyle, int accessory, int body) {
            this.id = id;
            this.name = name;
            this.role = role;
            this.perkName = perkName;
            this.perkDesc = perkDesc;
            this.hp = hp;
            this.power = power;
            this.mass = mass;
            this.speed = speed;
            this.dashCd = dashCd;
            this.perk = perk;
            this.skin = skin;
            this.hair = hair;
            this.top = top;
            this.bottom = bottom;
            this.accent = accent;
            this.shoe = shoe;
            this.hairStyle = hairStyle;
            this.accessory = accessory;
            this.body = body;
        }
    }

    public static final Data[] ALL = {
            new Data("clerk", "Офисный клерк", "Сотрудник года", "Дедлайн",
                    "+25% урона в рывке", 95, 1.00f, 1.00f, 1.00f, 1.35f, PERK_DASH,
                    0xFFF0C29A, 0xFF3A2A24, 0xFF5C6B8C, 0xFF2F3A4D, 0xFFD9555F, 0xFF2A2320,
                    0, 1, 0),
            new Data("barista", "Бариста", "Мастер латте", "Двойной эспрессо",
                    "Рывок заряжается на 30% быстрее", 90, 0.94f, 0.95f, 1.10f, 1.35f, PERK_CD,
                    0xFFE9B78E, 0xFF5A3527, 0xFF3E7C6B, 0xFF30323B, 0xFFE6D6B8, 0xFF2A2724,
                    4, 4, 0),
            new Data("student", "Студент", "Вечно на паре", "Халява",
                    "+25% здоровья", 125, 0.95f, 0.90f, 1.05f, 1.35f, PERK_NONE,
                    0xFFEFC4A0, 0xFF2C2320, 0xFF7B5EA7, 0xFF3B3F52, 0xFFF0C75E, 0xFF312B2A,
                    2, 0, 1),
            new Data("teamlead", "Тимлид", "Синергия", "Синергия",
                    "+12% силы всей команде", 100, 0.92f, 1.00f, 1.00f, 1.30f, PERK_SYNERGY,
                    0xFFF2C7A0, 0xFF33251F, 0xFF3C4256, 0xFF272B38, 0xFF6FD3C7, 0xFF211E1D,
                    1, 1, 0),
            new Data("trucker", "Дальнобойщик", "Король трассы", "Фура",
                    "+35% массы", 120, 1.10f, 1.35f, 0.85f, 1.40f, PERK_NONE,
                    0xFFE4AF84, 0xFF4A3126, 0xFFB8563E, 0xFF39435B, 0xFFE9D9B8, 0xFF2B2523,
                    5, 3, 3),
            new Data("babushka", "Бабушка", "Божий одуванчик", "Божий одуванчик",
                    "-25% получаемого урона", 130, 0.82f, 1.10f, 0.80f, 1.45f, PERK_ARMOR,
                    0xFFF0C8A8, 0xFFD8D3CC, 0xFF8E5A7A, 0xFF6B4257, 0xFFD9B44A, 0xFF3B3230,
                    3, 5, 1),
            new Data("courier", "Курьер", "Доставка за 15 минут", "Срочно",
                    "+20% скорости", 85, 0.90f, 0.85f, 1.30f, 1.35f, PERK_NONE,
                    0xFFEAB98F, 0xFF2F241F, 0xFFE0A33E, 0xFF2F333F, 0xFF3A3F4C, 0xFF2A2523,
                    0, 6, 1),
            new Data("trainer", "Фитнес-тренер", "Ещё один подход", "Жим лёжа",
                    "+25% силы", 105, 1.30f, 1.10f, 1.00f, 1.35f, PERK_NONE,
                    0xFFDDA477, 0xFF2A201B, 0xFF3E7FB5, 0xFF2D3038, 0xFFFFFFFF, 0xFF23201F,
                    5, 0, 2),
            new Data("gopnik", "Гопник", "С района", "С района",
                    "+20% массы и +10% силы", 115, 1.10f, 1.20f, 0.95f, 1.35f, PERK_NONE,
                    0xFFE7B088, 0xFF38261D, 0xFF2B2B33, 0xFF23232A, 0xFFE8E8EE, 0xFF1E1C1C,
                    1, 0, 2),
            new Data("teacher", "Учительница", "Тишина в классе", "Дисциплина",
                    "+15% силы и здоровья", 110, 1.05f, 0.95f, 1.00f, 1.35f, PERK_NONE,
                    0xFFF1C4A4, 0xFF6B4230, 0xFF9C5568, 0xFF4A3545, 0xFFE5D6BC, 0xFF322B2B,
                    3, 1, 0),
            new Data("dev", "Программист", "Легаси не спит", "Легаси-код",
                    "+30% здоровья, -10% скорости", 140, 0.96f, 1.00f, 0.88f, 1.35f, PERK_NONE,
                    0xFFE9BE9C, 0xFF2A2320, 0xFF49536B, 0xFF2B2F3A, 0xFF6FCF97, 0xFF2A2726,
                    4, 2, 1),
            new Data("blogger", "Блогер", "Лайк и подписка", "Хайп",
                    "+30% урона в рывке, -10% массы", 85, 1.00f, 0.80f, 1.22f, 1.35f, PERK_DASH,
                    0xFFF2C6A2, 0xFF26201E, 0xFFE0647A, 0xFF2B2635, 0xFF58C8D6, 0xFF2C2626,
                    2, 5, 1),
            new Data("guard", "Охранник", "Пропускной режим", "Прочный корпус",
                    "+20% массы", 120, 1.15f, 1.25f, 0.92f, 1.38f, PERK_NONE,
                    0xFFE8B58E, 0xFF2D2520, 0xFF2F4258, 0xFF232B36, 0xFFC9B15C, 0xFF1F1D1C,
                    0, 6, 3),
            new Data("poet", "Поэт", "Слово лечит", "Слово лечит",
                    "Команда лечится после боя", 90, 0.86f, 0.90f, 1.05f, 1.35f, PERK_REGEN,
                    0xFFEFC2A0, 0xFF3A2A22, 0xFF6C5B8E, 0xFF3B3348, 0xFFE0A8B5, 0xFF2E2828,
                    2, 0, 0),
    };

    private Chars() {
    }

    public static Data byId(String id) {
        for (Data d : ALL) {
            if (d.id.equals(id)) return d;
        }
        return ALL[0];
    }
}

