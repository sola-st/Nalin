"""

Created on 19-March-2020
@author Jibesh Patra


Simple AST based data_transformations to track assignment values during execution of python
programs.

"""
import libcst as cst
import libcst.matchers as matchers


class ReplaceUnnecessaryCode(cst.CSTTransformer):
    """
    Comment-out/Remove unnecessary code like plt.show() that keeps executing
    programs stuck
    """

    def leave_Decorator(self, node, updated_node):
        # Matches decorators of type @jit
        numba_jit_matcher = matchers.Decorator(decorator=matchers.Call(func=matchers.Name(value="jit")))
        if matchers.matches(node, numba_jit_matcher):
            return cst.EmptyLine()
        else:
            return node

    def leave_Call(self, node: cst.CSTNode, updated_node: cst.CSTNode) -> cst.CSTNode:
        # Match calls like plt.show()
        plt_show_matcher = matchers.Call(
            func=matchers.Attribute(value=matchers.Name('plt'), attr=matchers.Name('show')))

        print_matcher = matchers.Call(func=matchers.Name('print'))
        empty_print = cst.Call(func=cst.Name(value='print'))

        # Comment out all plt.show
        if matchers.matches(node, plt_show_matcher):
            # return updated_node.with_changes(func=cst.Attribute(value=cst.Name('plt'), attr=cst.Name('draw')))
            return cst.Pass()
        # Comment out prints
        # if matchers.matches(node, print_matcher):
        #     return empty_print
        return updated_node


class ModifyAssignmentsToTrackValues(cst.CSTTransformer):
    def __init__(self, python_file_path, out_dir='/home/dynamic_analysis_outputs'):
        super().__init__()
        self.file_path = python_file_path
        self.out_dir = out_dir  # The output directory path where the dynamic analysis results will be written

    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    def leave_SimpleStatementLine(self, node, updated_node):
        body = []
        for node_line in node.body:
            if isinstance(node_line, cst.Assign) or isinstance(node_line, cst.AugAssign):
                line_number = str(self.get_metadata(node=node, key=cst.metadata.PositionProvider).start.line)
                all_targets = []
                # TODO: We currently only consider assignments/augassigns of type a = b and ignore a.b.c = m or b[c] = m.
                if isinstance(node_line, cst.Assign):
                    # Go through all targets. Only an Assignment can have multiple targets. Eg. a=b=23 or a,b,c = 2,3,4
                    for assign_target in node_line.targets:
                        target_single_var = matchers.AssignTarget(target=matchers.Name())
                        if matchers.matches(assign_target, target_single_var):
                            all_targets.append(assign_target.target.value)
                        # Track values of type a,b,c = 1,2,3
                        target_tuple = matchers.AssignTarget(target=matchers.Tuple())
                        if matchers.matches(assign_target, target_tuple):
                            for elem in assign_target.children:
                                for v in elem.children:
                                    if matchers.matches(v, matchers.Element(value=matchers.Name())):
                                        all_targets.append(v.value.value)

                    if not len(all_targets):
                        return updated_node
                elif isinstance(node_line, cst.AugAssign):
                    target_single_var = matchers.AugAssign(target=matchers.Name())
                    if matchers.matches(node_line, target_single_var):
                        all_targets.append(node_line.target.value)
                    else:
                        return updated_node
                # One call for each target. Eg a = b = 23. A call each for 'a' & 'b'
                call_expr_nodes = []
                for target_var in all_targets:
                    # dc = cst.Dict([
                    #     cst.DictElement(cst.Name('variable_name'), cst.Name(target_var)),
                    #     cst.DictElement(cst.Name('line_number'), cst.Integer(line_number)),
                    #     cst.DictElement(cst.Name('value'), node_line.value)
                    # ]
                    # )
                    call_expr_node = cst.Expr(cst.Call(
                        cst.Name("MAGIC_DYNAMIC_analysis_value_collector"),
                        (
                            cst.Arg(
                                keyword=cst.Name("file_name"),
                                value=cst.SimpleString('"""{}"""'.format(self.file_path)),
                                whitespace_after_arg=cst.SimpleWhitespace(" "),
                            ),
                            cst.Arg(
                                keyword=cst.Name("line_number"),
                                value=cst.Integer(line_number),
                                whitespace_after_arg=cst.SimpleWhitespace(" "),
                            ),
                            cst.Arg(
                                keyword=cst.Name("var_name"),
                                value=cst.SimpleString('"""{}"""'.format(target_var)),
                                whitespace_after_arg=cst.SimpleWhitespace(" "),
                            ),
                            cst.Arg(
                                keyword=cst.Name("value"),
                                value=cst.Name(target_var),
                                whitespace_after_arg=cst.SimpleWhitespace(" "),
                            ),
                            cst.Arg(
                                keyword=cst.Name("outdir"),
                                value=cst.SimpleString('"""{}"""'.format(self.out_dir)),
                                whitespace_after_arg=cst.SimpleWhitespace(" "),
                            ),
                        ),
                        whitespace_after_func=cst.SimpleWhitespace(" "),
                        whitespace_before_args=cst.SimpleWhitespace(" "),
                    ))
                    call_expr_nodes.append(call_expr_node)
                body.append(node.body[0])
                for expr_node in call_expr_nodes:
                    body.append(expr_node)
            else:
                body.append(node_line)
        return updated_node.with_changes(body=body)


class AddImportOnTopToTrackResults(cst.CSTTransformer):
    """
    Add an import on top of each file that we may use to track the result.
    This is a package that has been created by us with the sole purpose of saving the
    results of the dynamic analysis.
    """

    def leave_Module(self, node, updated_node):
        body = list(node.body)
        import_node = cst.ImportFrom(
            module=cst.Attribute(value=cst.Name("dynamic_analysis_tracker_local_package"),
                                 attr=cst.Name('collect_assignment_values')),
            names=(cst.ImportAlias(cst.Name("MAGIC_DYNAMIC_analysis_value_collector")),)
        )
        empty_line = cst.EmptyLine()
        body.insert(0, import_node)
        body.insert(1, empty_line)
        return updated_node.with_changes(body=body)




class CheckIfAlreadyInstrumented(cst.CSTVisitor):
    def __init__(self):
        super().__init__()
        self.already_instrumented = False

    def visit_ImportFrom(self, node):
        if self.already_instrumented: return

        # The import statement we put on the top of every file
        import_node = matchers.ImportFrom(
            module=matchers.Attribute(value=matchers.Name("dynamic_analysis_tracker_local_package"),
                                      attr=matchers.Name('collect_assignment_values')),
            names=(matchers.ImportAlias(matchers.Name("MAGIC_DYNAMIC_analysis_value_collector")),)
        )
        if matchers.matches(node, import_node):
            self.already_instrumented = True


def instrument_given_file_multiprocessing(args):
    in_file_path, out_file_path, out_dir = args
    return instrument_given_file(in_file_path, out_file_path, out_dir)


def instrument_given_file(in_file_path, out_file_path, out_dir_execution_output):
    instrumented_and_written = False
    with open(in_file_path, 'r') as file:
        src = file.read()
    try:
        ast = cst.parse_module(src)
    except Exception as e:
        print(e)
        return False  # Syntax error

    try:
        # Check for the presence of
        check_if_already_instrumented = CheckIfAlreadyInstrumented()
        ast_checked_for_instrumentation = ast.visit(check_if_already_instrumented)
        if check_if_already_instrumented.already_instrumented:
            return False

        modify_assignments = ModifyAssignmentsToTrackValues(python_file_path=in_file_path,
                                                            out_dir=out_dir_execution_output)
        # Add line number information
        ast = cst.MetadataWrapper(ast)

        # Modify Assignments first else adding more code / deleting can mess uo the line numbers
        ast = ast.visit(modify_assignments)
        # print("Could not modify assignments for file {} ".format(in_file_path))

        # Replace code such as print and plt.show() etc. with  pass
        replace_unnecessary_code = ReplaceUnnecessaryCode()
        ast = ast.visit(replace_unnecessary_code)

        # Add a function that is called to save results
        track_result = AddImportOnTopToTrackResults()
        ast = ast.visit(track_result)
        instrumented_and_written = True
    except Exception as e:
        print(e, in_file_path)
        return instrumented_and_written

    # Finally, write out the code
    with open(out_file_path, 'w') as file:
        if instrumented_and_written:
            # print("Writing instrumented_and_written file --> {}".format(out_file_path))
            try:
                instrumented_code = ast.code
                file.write(instrumented_code)
            except Exception as e:
                print(e)
                instrumented_and_written = False
    return instrumented_and_written


if __name__ == '__main__':
    from pathlib import Path

    python_files = list(Path('benchmark/scripts').rglob('*.py'))
    for fl in python_files:
        file_path = str(fl)
        print('Instrumenting ', file_path)
        file_path_instrumented = 'benchmark/scripts/test_instrumented.py'
        instrument_given_file(in_file_path=file_path, out_file_path=file_path_instrumented,
                              out_dir_execution_output='/home/jibesh')
