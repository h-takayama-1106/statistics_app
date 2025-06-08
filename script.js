document.addEventListener('DOMContentLoaded', () => {
    const appContainer = document.getElementById('app-container');
    const addButton = document.getElementById('add-button');
    const MAX_ITEMS = 10;
    const STORAGE_KEY = 'simpleAppData';

    // ローカルストレージからデータを読み込み、表示する
    function loadItems() {
        const items = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
        items.forEach(item => addItemToDOM(item));
        updateAddButtonState();
    }

    // DOMに新しいアイテム（テキストボックスとボタン）を追加する
    function addItemToDOM(value = '') {
        if (appContainer.children.length >= MAX_ITEMS) {
            return;
        }

        const itemDiv = document.createElement('div');
        itemDiv.classList.add('item');

        const input = document.createElement('input');
        input.type = 'text';
        input.value = value;

        const saveButton = document.createElement('button');
        saveButton.textContent = '保存';
        saveButton.addEventListener('click', () => {
            saveItem(input.value);
            alert('保存完了しました');
            window.location.href = 'display.html'; // 保存後にdisplay.htmlに遷移
        });

        const deleteButton = document.createElement('button');
        deleteButton.textContent = '削除';
        deleteButton.addEventListener('click', () => {
            deleteItem(itemDiv, input.value);
        });

        itemDiv.appendChild(input);
        itemDiv.appendChild(saveButton);
        itemDiv.appendChild(deleteButton);
        appContainer.appendChild(itemDiv);

        updateAddButtonState();
    }

    // アイテムをローカルストレージに保存する
    function saveItem(value) {
        if (value.trim() === '') { // 空の文字列は保存しない
            return;
        }
        const items = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
        // 重複を避けるため、既存の同じ値は削除してから追加する
        const filteredItems = items.filter(item => item !== value);
        filteredItems.push(value);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(filteredItems));
    }

    // アイテムをDOMから削除し、ローカルストレージからも削除する
    function deleteItem(itemElement, value) {
        appContainer.removeChild(itemElement);
        const items = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
        const filteredItems = items.filter(item => item !== value);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(filteredItems));
        updateAddButtonState();
    }

    // 追加ボタンの状態を更新する（最大数に達したら無効化）
    function updateAddButtonState() {
        if (appContainer.children.length >= MAX_ITEMS) {
            addButton.disabled = true;
        } else {
            addButton.disabled = false;
        }
    }

    // イベントリスナーを設定
    addButton.addEventListener('click', () => {
        addItemToDOM();
    });

    // ページ読み込み時にアイテムを読み込む
    loadItems();
});
