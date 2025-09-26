import re
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from colors import Colors


class ExpertSystem:
    """
    Экспертная система с прямой цепочкой рассуждений
    для предметной области "Умный дом"
    """

    def __init__(self, rules_file: str = "rules.txt"):
        self.rules_file = rules_file
        self.rules = []
        self.facts = {}
        self.derived_facts = set()
        self.inference_log = []
        self.animation_speed = 0.05
        self.load_rules()
        self.initialize_facts()

    def clear_screen(self):
        """Очистка экрана"""
        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self, text: str, color=Colors.BRIGHT_CYAN):
        """Печать заголовка с рамкой"""
        width = 60
        border = "═" * width
        print(f"\n{color}╔{border}╗{Colors.RESET}")
        print(f"{color}║{text.center(width)}║{Colors.RESET}")
        print(f"{color}╚{border}╝{Colors.RESET}")

    def print_section(self, title: str, color=Colors.BRIGHT_YELLOW):
        """Печать секции"""
        print(f"\n{color}▓▓▓ {title} ▓▓▓{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * (len(title) + 8)}{Colors.RESET}")

    def print_success(self, message: str):
        """Печать сообщения об успехе"""
        print(f"{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")

    def print_warning(self, message: str):
        """Печать предупреждения"""
        print(f"{Colors.BRIGHT_YELLOW}⚠ {message}{Colors.RESET}")

    def print_error(self, message: str):
        """Печать ошибки"""
        print(f"{Colors.BRIGHT_RED}✗ {message}{Colors.RESET}")

    def print_info(self, message: str):
        """Печать информации"""
        print(f"{Colors.BRIGHT_BLUE}ℹ {message}{Colors.RESET}")

    def print_fact(self, key: str, value: str, is_derived: bool = False):
        """Печать факта с цветовой разметкой"""
        icon = "🔍" if is_derived else "📋"
        color = Colors.BRIGHT_GREEN if is_derived else Colors.WHITE
        print(f"  {icon} {color}{key}{Colors.RESET} = {Colors.BRIGHT_WHITE}{value}{Colors.RESET}")

    def animate_text(self, text: str, delay: float = None):
        """Анимированный вывод текста"""
        if delay is None:
            delay = self.animation_speed
        for char in text:
            print(char, end="", flush=True)
            time.sleep(delay)
        print()

    def print_progress_bar(self, current: int, total: int, description: str = ""):
        """Печать прогресс-бара"""
        percent = int((current / total) * 100)
        bar_length = 30
        filled_length = int(bar_length * current // total)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(
            f"\r{Colors.BRIGHT_BLUE}{description} [{bar}] {percent}%{Colors.RESET}",
            end="",
            flush=True,
        )

    def initialize_facts(self):
        """Инициализация стартовой ситуации"""
        self.facts = {
            "время_суток": "вечер",
            "день_недели": "рабочий",
            "присутствие_людей": "да",
            "температура_внешняя": "холодно",
            "освещенность": "темно",
        }

        self.print_section("Инициализация системы", Colors.BRIGHT_MAGENTA)
        self.animate_text("🏠 Загружаю параметры умного дома...")
        time.sleep(0.5)

        print(f"\n{Colors.BRIGHT_CYAN}📋 Стартовая ситуация:{Colors.RESET}")
        for key, value in self.facts.items():
            self.print_fact(key, value)

        print(f"\n{Colors.DIM}{'─' * 50}{Colors.RESET}")
        self.derived_facts.clear()
        self.inference_log.clear()

    def load_rules(self):
        """Загрузка правил из файла"""
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                self.rules = []
                rule_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        try:
                            rule = self.parse_rule(line)
                            if rule:
                                self.rules.append(rule)
                                rule_count += 1
                        except Exception as e:
                            self.print_error(f"Ошибка в строке {line_num}: {e}")

                if rule_count > 0:
                    self.print_success(f"Загружено правил: {rule_count}")
                else:
                    self.print_warning("Правила не найдены")

        except FileNotFoundError:
            self.print_warning(f"Файл {self.rules_file} не найден")
            self.animate_text("📝 Создаю новый файл с базовыми правилами...")
            self.create_default_rules()

    def create_default_rules(self):
        """Создание файла с базовыми правилами для умного дома"""
        default_rules = [
            "# Правила для системы освещения",
            "ЕСЛИ время_суток=вечер И присутствие_людей=да ТО включить_основное_освещение=да",
            "ЕСЛИ время_суток=ночь И движение_в_коридоре=да ТО включить_ночник=да",
            "ЕСЛИ освещенность=темно И присутствие_людей=да ТО включить_основное_освещение=да",
            "ЕСЛИ присутствие_людей=нет ТО выключить_все_освещение=да",
            "",
            "# Правила для системы отопления",
            "ЕСЛИ температура_внешняя=холодно И присутствие_людей=да ТО включить_отопление=да",
            "ЕСЛИ температура_внутренняя=жарко ТО уменьшить_отопление=да",
            "ЕСЛИ присутствие_людей=нет И день_недели=рабочий ТО режим_экономии_тепла=да",
            "",
            "# Правила безопасности",
            "ЕСЛИ присутствие_людей=нет И время_суток=день ТО включить_охрану=да",
            "ЕСЛИ движение_на_входе=да И включить_охрану=да ТО сигнал_тревоги=да",
            "ЕСЛИ дым=да ТО пожарная_тревога=да",
            "ЕСЛИ утечка_газа=да ТО перекрыть_газ=да",
            "",
            "# Правила для развлечений",
            "ЕСЛИ время_суток=вечер И день_недели=выходной И присутствие_людей=да ТО включить_развлекательную_систему=да",
            "ЕСЛИ включить_развлекательную_систему=да ТО приглушить_освещение=да",
            "",
            "# Правила для экономии энергии",
            "ЕСЛИ присутствие_людей=нет ТО режим_экономии_энергии=да",
            "ЕСЛИ режим_экономии_энергии=да ТО отключить_неприоритетные_устройства=да",
            "",
            "# Утренние правила",
            "ЕСЛИ время_суток=утро И день_недели=рабочий И присутствие_людей=да ТО включить_кофеварку=да",
            "ЕСЛИ время_суток=утро И присутствие_людей=да ТО включить_новости=да",
        ]

        with open(self.rules_file, "w", encoding="utf-8") as f:
            f.write("\n".join(default_rules))
        self.load_rules()

    def parse_rule(self, rule_text: str) -> Optional[Dict]:
        """Парсинг правила вида: ЕСЛИ условие ТО заключение"""
        rule_text = " ".join(rule_text.split())

        pattern = r"ЕСЛИ\s+(.+?)\s+ТО\s+(.+)"
        match = re.match(pattern, rule_text, re.IGNORECASE)

        if not match:
            return None

        conditions_str = match.group(1)
        conclusion_str = match.group(2)

        conditions = []
        condition_parts = re.split(r"\s+И\s+", conditions_str, flags=re.IGNORECASE)

        for part in condition_parts:
            cond_match = re.match(r"(\w+)\s*=\s*(.+)", part.strip())
            if cond_match:
                obj = cond_match.group(1).strip()
                value = cond_match.group(2).strip()
                conditions.append((obj, value))

        concl_match = re.match(r"(\w+)\s*=\s*(.+)", conclusion_str.strip())
        if not concl_match:
            return None

        conclusion_obj = concl_match.group(1).strip()
        conclusion_value = concl_match.group(2).strip()

        return {
            "conditions": conditions,
            "conclusion": (conclusion_obj, conclusion_value),
            "text": rule_text,
        }

    def check_rule_conditions(self, rule: Dict) -> bool:
        """Проверка выполнения условий правила"""
        for obj, value in rule["conditions"]:
            if obj not in self.facts or self.facts[obj] != value:
                return False
        return True

    def apply_rule(self, rule: Dict) -> bool:
        """Применение правила (добавление нового факта)"""
        conclusion_obj, conclusion_value = rule["conclusion"]

        if conclusion_obj not in self.facts:
            self.facts[conclusion_obj] = conclusion_value
            self.derived_facts.add(conclusion_obj)

            print(f"\n{Colors.BRIGHT_GREEN}⚡ Правило сработало!{Colors.RESET}")
            print(f"{Colors.DIM}┌─ Условие: {Colors.RESET}{self._format_conditions(rule['conditions'])}")
            print(
                f"{Colors.DIM}└─ Вывод: {Colors.RESET}{Colors.BRIGHT_YELLOW}{conclusion_obj} = {conclusion_value}{Colors.RESET}"
            )

            self.inference_log.append(
                {
                    "rule": rule["text"],
                    "conclusion": f"{conclusion_obj} = {conclusion_value}",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                }
            )

            time.sleep(0.5)
            return True

        return False

    def _format_conditions(self, conditions: List[Tuple[str, str]]) -> str:
        """Форматирование условий для вывода"""
        formatted = []
        for obj, value in conditions:
            formatted.append(f"{Colors.CYAN}{obj}={value}{Colors.RESET}")
        return f" {Colors.WHITE}И{Colors.RESET} ".join(formatted)

    def forward_chaining(self) -> List[str]:
        """Прямая цепочка рассуждений"""
        self.print_section("Механизм логического вывода", Colors.BRIGHT_BLUE)
        self.animate_text("🧠 Запускаю анализ правил...")

        applied_rules = []
        changed = True
        iteration = 1

        while changed:
            changed = False

            if iteration > 1:
                print(f"\n{Colors.BRIGHT_MAGENTA}🔄 Итерация {iteration}:{Colors.RESET}")
            else:
                print(f"\n{Colors.BRIGHT_MAGENTA}🔍 Поиск применимых правил:{Colors.RESET}")

            for i, rule in enumerate(self.rules):
                self.print_progress_bar(i + 1, len(self.rules), "Анализирую правила")
                time.sleep(0.02)

                if self.check_rule_conditions(rule):
                    if self.apply_rule(rule):
                        applied_rules.append(rule["text"])
                        changed = True

            print()

            if not changed:
                if iteration == 1:
                    self.print_info("Новых применимых правил не найдено")
                else:
                    self.print_success("Все возможные выводы сделаны")

            iteration += 1

            if iteration > 10:
                self.print_warning("Достигнуто максимальное количество итераций")
                break

        return applied_rules

    def ask_user_for_facts(self) -> bool:
        """Запрос у пользователя дополнительных фактов"""
        self.print_section("Требуется дополнительная информация", Colors.BRIGHT_YELLOW)

        print(f"{Colors.BRIGHT_YELLOW}🤔 Не удается применить ни одно правило{Colors.RESET}")
        print(f"{Colors.DIM}Для продолжения работы нужна дополнительная информация{Colors.RESET}")

        possible_facts = [
            ("движение_в_коридоре", "да/нет", "Обнаружено движение в коридоре"),
            ("движение_на_входе", "да/нет", "Обнаружено движение на входе"),
            ("температура_внутренняя", "жарко/нормально/холодно", "Температура в доме"),
            ("дым", "да/нет", "Обнаружен дым"),
            ("утечка_газа", "да/нет", "Обнаружена утечка газа"),
            ("время_суток", "утро/день/вечер/ночь", "Время суток"),
            ("день_недели", "рабочий/выходной", "Тип дня"),
        ]

        print(f"\n{Colors.BRIGHT_BLUE}💡 Возможные параметры для ввода:{Colors.RESET}")
        for i, (fact, values, description) in enumerate(possible_facts, 1):
            if fact not in self.facts:
                print(f"  {Colors.BRIGHT_MAGENTA}{i}.{Colors.RESET} {Colors.CYAN}{fact}{Colors.RESET}")
                print(f"     {Colors.DIM}{description} ({values}){Colors.RESET}")

        print(f"\n{Colors.DIM}Введите параметр в формате: 'название=значение'{Colors.RESET}")
        print(f"{Colors.DIM}Или наберите 'стоп' для завершения{Colors.RESET}")

        while True:
            try:
                user_input = input(f"\n{Colors.BRIGHT_WHITE}➤ {Colors.RESET}").strip()

                if user_input.lower() in ["стоп", "stop", "exit", "quit"]:
                    self.print_info("Работа завершена пользователем")
                    return False

                match = re.match(r"(\w+)\s*=\s*(.+)", user_input)
                if match:
                    obj = match.group(1).strip()
                    value = match.group(2).strip()
                    self.facts[obj] = value
                    self.print_success(f"Добавлен факт: {obj} = {value}")
                    return True
                else:
                    self.print_error("Неверный формат. Используйте: 'название=значение'")

            except KeyboardInterrupt:
                print(f"\n{Colors.BRIGHT_YELLOW}Работа прервана пользователем{Colors.RESET}")
                return False

    def show_inference_log(self):
        """Показать лог вывода"""
        if not self.inference_log:
            self.print_info("Лог пуст - правила еще не применялись")
            return

        self.print_section("Журнал логического вывода", Colors.BRIGHT_MAGENTA)

        for i, entry in enumerate(self.inference_log, 1):
            print(f"\n{Colors.BRIGHT_MAGENTA}#{i} [{entry['timestamp']}]{Colors.RESET}")
            print(f"{Colors.DIM}├─ Правило:{Colors.RESET} {entry['rule']}")
            print(
                f"{Colors.DIM}└─ Вывод:{Colors.RESET} {Colors.BRIGHT_GREEN}{entry['conclusion']}{Colors.RESET}"
            )

    def show_system_recommendations(self):
        """Показать рекомендации системы"""
        if not self.derived_facts:
            self.print_info("Пока нет рекомендаций")
            return

        self.print_section("Рекомендации умного дома", Colors.BRIGHT_GREEN)

        categories = {
            "Освещение": [
                "включить_основное_освещение",
                "включить_ночник",
                "приглушить_освещение",
                "выключить_все_освещение",
            ],
            "Климат": [
                "включить_отопление",
                "уменьшить_отопление",
                "режим_экономии_тепла",
                "оптимальная_температура",
            ],
            "Безопасность": [
                "включить_охрану",
                "сигнал_тревоги",
                "пожарная_тревога",
                "перекрыть_газ",
            ],
            "Развлечения": ["включить_развлекательную_систему", "включить_новости"],
            "Энергосбережение": [
                "режим_экономии_энергии",
                "отключить_неприоритетные_устройства",
            ],
            "Утренние процедуры": ["включить_кофеварку"],
            "Дополнительно": ["ночной_режим", "приглушить_все_звуки"],
        }

        icons = {
            "Освещение": "💡",
            "Климат": "🌡️",
            "Безопасность": "🔒",
            "Развлечения": "📺",
            "Энергосбережение": "⚡",
            "Утренние процедуры": "☕",
            "Дополнительно": "⚙️",
        }

        for category, facts_list in categories.items():
            category_facts = [f for f in facts_list if f in self.derived_facts and f in self.facts]
            if category_facts:
                icon = icons.get(category, "📋")
                print(f"\n{icon} {Colors.BRIGHT_YELLOW}{category}:{Colors.RESET}")
                for fact in category_facts:
                    value = self.facts[fact]
                    if value == "да":
                        print(f"  {Colors.BRIGHT_GREEN}✓{Colors.RESET} {fact.replace('_', ' ').title()}")
                    else:
                        print(
                            f"  {Colors.BRIGHT_BLUE}•{Colors.RESET} {fact.replace('_', ' ').title()}: {value}"
                        )

    def run(self):
        """Основной цикл работы экспертной системы"""
        self.clear_screen()
        self.print_header("ЭКСПЕРТНАЯ СИСТЕМА 'УМНЫЙ ДОМ'")

        while True:
            applied_rules = self.forward_chaining()

            self.display_facts()

            if applied_rules:
                self.show_system_recommendations()

                can_apply_more = False
                for rule in self.rules:
                    if self.check_rule_conditions(rule):
                        conclusion_obj, _ = rule["conclusion"]
                        if conclusion_obj not in self.facts:
                            can_apply_more = True
                            break

                if not can_apply_more:
                    self.print_section("Анализ завершен", Colors.BRIGHT_GREEN)
                    self.print_success("Все возможные выводы сделаны")

                    print(f"\n{Colors.BRIGHT_BLUE}Доступные действия:{Colors.RESET}")
                    print(f"  1. {Colors.CYAN}Показать журнал вывода{Colors.RESET}")
                    print(f"  2. {Colors.CYAN}Добавить новые факты{Colors.RESET}")
                    print(f"  3. {Colors.CYAN}Завершить работу{Colors.RESET}")

                    choice = input(f"\n{Colors.BRIGHT_WHITE}Выберите действие (1-3): {Colors.RESET}").strip()

                    if choice == "1":
                        self.show_inference_log()
                        input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
                    elif choice == "2":
                        if self.ask_user_for_facts():
                            continue
                        else:
                            break
                    else:
                        break
            else:
                if not self.ask_user_for_facts():
                    break

        self.print_section("Сеанс завершен", Colors.BRIGHT_MAGENTA)
        self.animate_text("🏠 Спасибо за использование системы умного дома!")

    def edit_rules_menu(self):
        """Меню редактирования правил"""
        while True:
            self.clear_screen()
            self.print_header("РЕДАКТИРОВАНИЕ БАЗЫ ПРАВИЛ", Colors.BRIGHT_YELLOW)

            print(f"\n{Colors.BRIGHT_BLUE}📝 Доступные действия:{Colors.RESET}")
            print(f"  {Colors.BRIGHT_CYAN}1.{Colors.RESET} Показать все правила")
            print(f"  {Colors.BRIGHT_CYAN}2.{Colors.RESET} Добавить новое правило")
            print(f"  {Colors.BRIGHT_CYAN}3.{Colors.RESET} Удалить правило")
            print(f"  {Colors.BRIGHT_CYAN}4.{Colors.RESET} Импорт правил из файла")
            print(f"  {Colors.BRIGHT_CYAN}5.{Colors.RESET} Экспорт правил в файл")
            print(f"  {Colors.BRIGHT_CYAN}6.{Colors.RESET} Вернуться в главное меню")

            choice = input(f"\n{Colors.BRIGHT_WHITE}➤ Выберите действие (1-6): {Colors.RESET}").strip()

            if choice == "1":
                self.show_rules()
            elif choice == "2":
                self.add_rule()
            elif choice == "3":
                self.delete_rule()
            elif choice == "4":
                self.import_rules()
            elif choice == "5":
                self.export_rules()
            elif choice == "6":
                break
            else:
                self.print_error("Неверный выбор. Введите число от 1 до 6")

            if choice != "6":
                input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")

    def show_rules(self):
        """Отображение всех правил"""
        self.print_section("Текущие правила", Colors.BRIGHT_CYAN)

        if not self.rules:
            self.print_warning("База правил пуста")
            return

        for i, rule in enumerate(self.rules, 1):
            print(f"\n{Colors.BRIGHT_MAGENTA}{i:2d}.{Colors.RESET} {rule['text']}")

            conditions_str = self._format_conditions(rule["conditions"])
            conclusion_obj, conclusion_value = rule["conclusion"]

            print(f"    {Colors.DIM}├─ Условия: {conditions_str}")
            print(f"    └─ Вывод: {Colors.CYAN}{conclusion_obj}={conclusion_value}{Colors.RESET}")

        print(f"\n{Colors.BRIGHT_GREEN}Всего правил: {len(self.rules)}{Colors.RESET}")

    def add_rule(self):
        """Добавление нового правила"""
        self.print_section("Добавление правила", Colors.BRIGHT_GREEN)

        print(f"{Colors.BRIGHT_BLUE}📝 Формат правила:{Colors.RESET}")
        print(f"  {Colors.DIM}ЕСЛИ условие ТО заключение{Colors.RESET}")
        print(f"  {Colors.DIM}ЕСЛИ условие1 И условие2 ТО заключение{Colors.RESET}")

        print(f"\n{Colors.BRIGHT_YELLOW}💡 Пример:{Colors.RESET}")
        print(
            f"  {Colors.CYAN}ЕСЛИ температура=жарко И кондиционер=выключен ТО включить_кондиционер=да{Colors.RESET}"
        )

        rule_text = input(f"\n{Colors.BRIGHT_WHITE}➤ Введите правило: {Colors.RESET}").strip()

        if not rule_text:
            self.print_error("Правило не может быть пустым")
            return

        rule = self.parse_rule(rule_text)

        if rule:
            self.rules.append(rule)
            self.save_rules()
            self.print_success("Правило добавлено успешно")

            print(f"\n{Colors.BRIGHT_BLUE}Анализ правила:{Colors.RESET}")
            conditions_str = self._format_conditions(rule["conditions"])
            conclusion_obj, conclusion_value = rule["conclusion"]
            print(f"  {Colors.DIM}Условия: {conditions_str}")
            print(f"  Вывод: {Colors.CYAN}{conclusion_obj}={conclusion_value}{Colors.RESET}")
        else:
            self.print_error("Ошибка в формате правила")

    def delete_rule(self):
        """Удаление правила"""
        if not self.rules:
            self.print_warning("База правил пуста")
            return

        self.show_rules()

        try:
            rule_num = input(
                f"\n{Colors.BRIGHT_WHITE}➤ Введите номер правила для удаления (1-{len(self.rules)}): {Colors.RESET}"
            ).strip()

            if rule_num.lower() == "отмена":
                self.print_info("Удаление отменено")
                return

            rule_num = int(rule_num) - 1

            if 0 <= rule_num < len(self.rules):
                deleted_rule = self.rules[rule_num]

                print(f"\n{Colors.BRIGHT_RED}⚠ Удаляемое правило:{Colors.RESET}")
                print(f"  {deleted_rule['text']}")

                confirm = (
                    input(f"\n{Colors.BRIGHT_YELLOW}Подтвердите удаление (да/нет): {Colors.RESET}")
                    .strip()
                    .lower()
                )

                if confirm in ["да", "yes", "y"]:
                    self.rules.pop(rule_num)
                    self.save_rules()
                    self.print_success("Правило удалено")
                else:
                    self.print_info("Удаление отменено")
            else:
                self.print_error("Неверный номер правила")

        except ValueError:
            self.print_error("Введите корректный номер")
        except KeyboardInterrupt:
            self.print_info("\nОперация отменена")

    def import_rules(self):
        """Импорт правил из файла"""
        filename = input(f"{Colors.BRIGHT_WHITE}➤ Введите имя файла: {Colors.RESET}").strip()

        try:
            with open(filename, "r", encoding="utf-8") as f:
                imported_rules = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith("#"):
                        rule = self.parse_rule(line)
                        if rule:
                            imported_rules.append(rule)
                        else:
                            self.print_warning(f"Пропущена строка {line_num}: неверный формат")

                if imported_rules:
                    self.rules.extend(imported_rules)
                    self.save_rules()
                    self.print_success(f"Импортировано правил: {len(imported_rules)}")
                else:
                    self.print_warning("Не найдено валидных правил для импорта")

        except FileNotFoundError:
            self.print_error("Файл не найден")
        except Exception as e:
            self.print_error(f"Ошибка при импорте: {e}")

    def export_rules(self):
        """Экспорт правил в файл"""
        filename = input(f"{Colors.BRIGHT_WHITE}➤ Введите имя файла для экспорта: {Colors.RESET}").strip()

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Экспорт правил из системы 'Умный дом'\n")
                f.write(f"# Дата экспорта: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n\n")

                for rule in self.rules:
                    f.write(rule["text"] + "\n")

            self.print_success(f"Правила экспортированы в файл: {filename}")

        except Exception as e:
            self.print_error(f"Ошибка при экспорте: {e}")

    def save_rules(self):
        """Сохранение правил в файл"""
        with open(self.rules_file, "w", encoding="utf-8") as f:
            f.write("# Правила экспертной системы 'Умный дом'\n\n")
            for rule in self.rules:
                f.write(rule["text"] + "\n")

    def display_facts(self):
        """Вывод текущих фактов (исходных и выведенных)"""
        self.print_section("Текущие факты", Colors.BRIGHT_CYAN)
        if not self.facts:
            self.print_warning("Фактов нет")
            return

        for key, value in self.facts.items():
            is_derived = key in self.derived_facts
            self.print_fact(key, value, is_derived)
