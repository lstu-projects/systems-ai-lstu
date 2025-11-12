import time

from colors import Colors
from expert_system_backward import BackwardExpertSystem


def main():
    """Главная функция"""
    system = BackwardExpertSystem()

    while True:
        system.clear_screen()
        system.print_header("ГЛАВНОЕ МЕНЮ", Colors.BRIGHT_CYAN)

        print(f"\n{Colors.DIM}┌─ Статистика системы ─────────────────────────────┐{Colors.RESET}")
        print(
            f"{Colors.DIM}│{Colors.RESET} Правил в базе: {Colors.BRIGHT_YELLOW}{len(system.rules):<26}{Colors.DIM}│{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}│{Colors.RESET} Известных фактов: {Colors.BRIGHT_GREEN}{len(system.facts):<23}{Colors.DIM}│{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}│{Colors.RESET} Доказано целей: {Colors.BRIGHT_BLUE}{len(system.inference_log):<25}{Colors.DIM}│{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}│{Colors.RESET} Файл правил: {Colors.BRIGHT_WHITE}{system.rules_file:<28}{Colors.DIM}│{Colors.RESET}"
        )
        print(f"{Colors.DIM}└──────────────────────────────────────────────────┘{Colors.RESET}")

        print(f"\n{Colors.BRIGHT_BLUE}🏠 Выберите действие:{Colors.RESET}")
        print(
            f"  {Colors.BRIGHT_CYAN}1.{Colors.RESET} {Colors.BRIGHT_GREEN}🚀 Запустить систему (доказать цель){Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_CYAN}2.{Colors.RESET} {Colors.BRIGHT_BLUE}📋 Показать текущие факты{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_CYAN}3.{Colors.RESET} {Colors.BRIGHT_MAGENTA}🔄 Сбросить к стартовому состоянию{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_CYAN}4.{Colors.RESET} {Colors.BRIGHT_WHITE}📊 Показать журнал вывода{Colors.RESET}"
        )
        print(
            f"  {Colors.BRIGHT_CYAN}5.{Colors.RESET} {Colors.BRIGHT_YELLOW}📚 Показать все правила{Colors.RESET}"
        )
        print(f"  {Colors.BRIGHT_CYAN}6.{Colors.RESET} {Colors.BRIGHT_RED}🚪 Выйти{Colors.RESET}")

        try:
            choice = input(f"\n{Colors.BRIGHT_WHITE}➤ Ваш выбор (1-6): {Colors.RESET}").strip()

            if choice == "1":
                system.run()
            elif choice == "2":
                system.display_facts()
                input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
            elif choice == "3":
                print(f"\n{Colors.BRIGHT_YELLOW}🔄 Сброс к стартовому состоянию...{Colors.RESET}")
                system.initialize_facts()
                system.print_success("Система сброшена к начальному состоянию")
                input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
            elif choice == "4":
                system.show_inference_log()
                input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
            elif choice == "5":
                show_all_rules(system)
                input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")
            elif choice == "6":
                system.print_section("До свидания!", Colors.BRIGHT_MAGENTA)
                system.animate_text("🏠 Благодарим за использование системы умного дома!")
                break
            else:
                system.print_error("Неверный выбор. Введите число от 1 до 6")
                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n{Colors.BRIGHT_YELLOW}Работа прервана пользователем{Colors.RESET}")
            break
        except Exception as e:
            system.print_error(f"Произошла ошибка: {e}")
            input(f"\n{Colors.DIM}Нажмите Enter для продолжения...{Colors.RESET}")


def show_all_rules(system):
    """Показать все правила в базе знаний"""
    system.print_section("База правил", Colors.BRIGHT_CYAN)

    if not system.rules:
        system.print_warning("База правил пуста")
        return

    categories = {
        "Освещение": [],
        "Отопление": [],
        "Безопасность": [],
        "Развлечения": [],
        "Энергосбережение": [],
        "Утренние процедуры": [],
        "Прочие": [],
    }

    for rule in system.rules:
        concl_obj, _ = rule["conclusion"]
        if "освещение" in concl_obj or "ночник" in concl_obj:
            categories["Освещение"].append(rule)
        elif "отопление" in concl_obj or "тепл" in concl_obj:
            categories["Отопление"].append(rule)
        elif "охран" in concl_obj or "тревог" in concl_obj or "газ" in concl_obj:
            categories["Безопасность"].append(rule)
        elif "развлекательн" in concl_obj or "новости" in concl_obj:
            categories["Развлечения"].append(rule)
        elif "энергии" in concl_obj or "устройства" in concl_obj:
            categories["Энергосбережение"].append(rule)
        elif "кофеварк" in concl_obj:
            categories["Утренние процедуры"].append(rule)
        else:
            categories["Прочие"].append(rule)

    icons = {
        "Освещение": "💡",
        "Отопление": "🌡️",
        "Безопасность": "🔒",
        "Развлечения": "📺",
        "Энергосбережение": "⚡",
        "Утренние процедуры": "☕",
        "Прочие": "📋",
    }

    for category, rules in categories.items():
        if rules:
            icon = icons.get(category, "📋")
            print(f"\n{icon} {Colors.BRIGHT_YELLOW}{category}:{Colors.RESET}")
            for i, rule in enumerate(rules, 1):
                print(f"  {Colors.BRIGHT_MAGENTA}{i}.{Colors.RESET} {rule['text']}")

    print(f"\n{Colors.BRIGHT_GREEN}Всего правил: {len(system.rules)}{Colors.RESET}")


if __name__ == "__main__":
    main()
