const theme_toggler = document.getElementById('theme-toggler');
const theme_icon = document.getElementById('theme-icon');
const icon_light = 'fa-sun';
const icon_dark = 'fa-moon';

const DARK_MODE_CLASS = 'dark-mode';
const DARK_MODE_STORAGE_KEY = 'dark-mode';

const toggle_dark_mode = (enable) => {
    if (enable) {
        document.body.classList.add(DARK_MODE_CLASS);
    } else {
        document.body.classList.remove(DARK_MODE_CLASS);
    }

    theme_icon.classList.remove(enable ? icon_dark : icon_light);
    theme_icon.classList.add(enable ? icon_light : icon_dark);

    localStorage.setItem(DARK_MODE_STORAGE_KEY, enable);
}

const should_enable_dark_mode = () => {
    // check the os preference, if we can't find it in the local storage
    if (localStorage.getItem(DARK_MODE_STORAGE_KEY) === null) {
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return true;
        }
        return false;
    }

    return localStorage.getItem(DARK_MODE_STORAGE_KEY) === 'true';
}


theme_toggler.addEventListener('click', function() {
    toggle_dark_mode(!document.body.classList.contains(DARK_MODE_CLASS));
});

const on_page_load = () => {
    toggle_dark_mode(should_enable_dark_mode());
}

on_page_load();