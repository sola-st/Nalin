let selected_files, JSON_data = null, required_threshold_distance = -1, table;

function triggerOnSelectingJSONFile(selection) {
    if (table)
        table.destroy();
    if (!selected_files)
        selected_files = selection.target.files;
    let f = selected_files[0];
    let reader = new FileReader();
    reader.onload = event => parseResult(JSON.parse(event.target.result));
    reader.readAsText(f);
}

function filter_data(data) {
    let filtered_data = [];
    if (required_threshold_distance < 0) return [];
    for (let k of Object.keys(data)) {

    }

    return filtered_data;
}

function radio_selection(current) {
    $('#inputJSON').prop('disabled', false);
    required_threshold_distance = parseFloat(current.value);
    if (table) {
        triggerOnSelectingJSONFile();
    }
}

function parseResult(data) {
    // data = filter_data(data);
    table = $('#example').DataTable({
        data: data,
        select: true,
        columns: [
            {title: 'file', data: 'file'},
            {title: 'var', data: 'var'},
            {title: 'line', data: 'line'},
            {title: 'value', data: 'value'},
            {title: 'type', data: 'type'},
            {title: 'len', data: 'len'},
            {title: 'p_buggy', data: 'predicted_p_buggy'},
        ],
        // order: [[5, "asc"]]
    });

    // Add
    // COLOR
    // table.rows().every(function (rowIdx, tableLoop, rowLoop) {
    //     let cell = table.cell({row: rowIdx, column: 1}).node();
    //     $(cell).addClass('fix');
    //
    //     cell = table.cell({row: rowIdx, column: 2}).node();
    //     $(cell).addClass('fix');
    //
    //
    //     cell = table.cell({row: rowIdx, column: 3}).node();
    //     $(cell).addClass('buggy');
    //
    //
    //     cell = table.cell({row: rowIdx, column: 4}).node();
    //     $(cell).addClass('buggy');
    //
    //
    // });
    //
    table
        .on('select', function (e, dt, type, indexes) {
            let rowData = table.rows(indexes).data().toArray()[0];
            let file_complete_path = rowData.file;
            let new_script_path = null;

            if (file_complete_path.includes('/python_repos/')) { // GitHub repos
                new_script_path = 'python_repos/' + file_complete_path.split('/python_repos/')[1];
            } else { // Jupyter notebooks
                new_script_path = "ipynb_scripts/" + rowData.file + '.py';
            }

            // let reference_info_text = `Current threshold distance: ${required_threshold_distance} ----- Distances: ${rowData.similarities_of_labels} ----- Num of Idfs, Lits ${rowData.num_of_available_identifiers}, ${rowData.num_of_available_literals} ----- Label Selection Way: ${rowData.type_of_label_selection}`;
            document.querySelector('.info h4').firstChild.nodeValue = `File Path : ${new_script_path}`;
            let lineNumber = rowData.line;

            let pre = document.querySelector('#pycode-original');
            let code = document.createElement('code');
            code.className = 'language-py';

            pre.textContent = '';
            code.textContent = 'Loading Python file ' + new_script_path;

            pre.appendChild(code);
            pre.setAttribute("data-line", lineNumber);

            $.get(new_script_path, function (data, textStatus, jqxhr) {
                code.textContent = data;
            }, 'text').done(function () {
                Prism.highlightElement(code);
                pre.setAttribute("data-line", lineNumber);
                pre.setAttribute('data-src-loaded', '');
                let highlightedElement = document.querySelector('#pycode-original .line-highlight');
                highlightedElement.scrollIntoView({block: 'center'});
            });

        });

}

function readDir(pathOfDir) {

}

document.getElementById('inputJSON').addEventListener('change', triggerOnSelectingJSONFile, false);
